"""LLM question categorisation."""
import json

import yaml

from ctxlayer.evaluation.categorize import CATEGORIES, categorize
from ctxlayer.evaluation.questions import Contestant, Question
from fakes import FakeLLM


def _question(number: int, text: str, sql: str | None = "SELECT 1") -> Question:
    contestant = Contestant(sql=sql, answer=f"SECRET-ANSWER-{number}", time_ms=100)
    return Question(qid=f"q{number:03d}", text=text, baselines={"reference": contestant})


def _reply(*items: tuple[str, str, float]) -> str:
    return "```json\n" + json.dumps([{"qid": q, "category": c, "confidence": p} for q, c, p in items]) + "\n```"


def test_categorize_writes_yaml_with_llm_categories(tmp_path):
    questions = [_question(1, "What is the latest value?"), _question(2, "Thanks!", sql=None)]
    llm = FakeLLM({"categorize": _reply(("q001", "lookup", 0.9), ("q002", "chit_chat", 0.8))})
    out = tmp_path / "eval" / "categories.yaml"

    result = categorize(questions, llm, out)

    expected = {
        "q001": {"question": "What is the latest value?", "category": "lookup",
                 "confidence": 0.9, "has_sql": True, "needs_review": False, "source": "llm"},
        "q002": {"question": "Thanks!", "category": "chit_chat",
                 "confidence": 0.8, "has_sql": False, "needs_review": False, "source": "llm"},
    }
    assert result == expected
    assert yaml.safe_load(out.read_text()) == expected


def test_categorize_prompt_has_tag_ids_text_and_has_sql_but_no_answers_or_sql(tmp_path):
    questions = [_question(1, "What is the latest value?", sql="SELECT secret_col FROM secret_tbl")]
    llm = FakeLLM({"categorize": _reply(("q001", "lookup", 0.9))})

    categorize(questions, llm, tmp_path / "c.yaml")
    (prompt, tools), = llm.calls

    assert prompt.splitlines()[0] == "[task: categorize]"
    assert "q001" in prompt
    assert "What is the latest value?" in prompt
    assert "has_sql" in prompt
    assert "SECRET-ANSWER" not in prompt
    assert "secret_tbl" not in prompt
    assert all(category in prompt for category in CATEGORIES)


def test_categorize_keeps_human_entry_untouched_and_skips_it_in_the_prompt(tmp_path):
    out = tmp_path / "categories.yaml"
    human = {"question": "old text", "category": "drivers", "confidence": 1.0,
             "has_sql": False, "needs_review": False, "source": "human"}
    out.write_text(yaml.safe_dump({"q001": human}))
    questions = [_question(1, "Why did it move?"), _question(2, "Latest value?")]
    llm = FakeLLM({"categorize": _reply(("q001", "lookup", 0.5), ("q002", "lookup", 0.7))})

    result = categorize(questions, llm, out)

    assert result["q001"] == human
    assert result["q002"]["category"] == "lookup"
    assert yaml.safe_load(out.read_text())["q001"] == human
    assert "q001" not in llm.calls[0][0]


def test_categorize_flags_invalid_category_for_review_and_defaults_to_lookup(tmp_path):
    questions = [_question(1, "Odd question")]
    llm = FakeLLM({"categorize": _reply(("q001", "astrology", 0.9))})

    result = categorize(questions, llm, tmp_path / "c.yaml")

    assert result["q001"]["category"] == "lookup"
    assert result["q001"]["needs_review"] is True
    assert result["q001"]["source"] == "llm"


def test_categorize_flags_every_question_when_reply_is_not_json(tmp_path):
    questions = [_question(1, "Q one")]
    llm = FakeLLM({"categorize": "I cannot do that."})

    result = categorize(questions, llm, tmp_path / "c.yaml")

    assert result["q001"]["category"] == "lookup"
    assert result["q001"]["needs_review"] is True


def test_categorize_sends_one_prompt_per_batch(tmp_path):
    questions = [_question(number, f"Question {number}") for number in range(1, 6)]
    batches = [
        _reply(("q001", "lookup", 0.9), ("q002", "drivers", 0.9)),
        _reply(("q003", "comparison", 0.9), ("q004", "volatility", 0.9)),
        _reply(("q005", "projection", 0.9)),
    ]
    llm = FakeLLM({"categorize": batches})

    result = categorize(questions, llm, tmp_path / "c.yaml", batch_size=2)

    assert len(llm.calls) == 3
    assert "q003" in llm.calls[1][0] and "q001" not in llm.calls[1][0]
    assert [result[q.qid]["category"] for q in questions] == [
        "lookup", "drivers", "comparison", "volatility", "projection",
    ]


def test_categorize_flags_questions_the_reply_leaves_out_or_cannot_assign(tmp_path):
    questions = [_question(1, "Q one"), _question(2, "Q two"), _question(3, "Q three")]
    items = [
        {"qid": "q001", "category": "projection", "confidence": 0.6},
        {"qid": ["q002"], "category": "lookup", "confidence": 0.9},  # unusable qid
        "junk",
        {"qid": "q099", "category": "drivers", "confidence": 0.9},  # never requested
    ]
    llm = FakeLLM({"categorize": json.dumps(items)})

    result = categorize(questions, llm, tmp_path / "c.yaml")

    assert list(result) == ["q001", "q002", "q003"]
    assert (result["q001"]["category"], result["q001"]["needs_review"]) == ("projection", False)
    assert (result["q002"]["category"], result["q002"]["needs_review"]) == ("lookup", True)
    assert (result["q003"]["category"], result["q003"]["needs_review"]) == ("lookup", True)
