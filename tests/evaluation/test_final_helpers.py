"""Question windows, citation validity, driver checklist and quality checks."""
import json
from datetime import date

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import write_yaml
from ctxlayer.evaluation.citations import citation_validity, driver_checklist_score
from ctxlayer.evaluation.questions import Contestant, Question, question_windows
from ctxlayer.evaluation.quality import quality_checks
from fakes import FakeLLM, per_citation

NOW = date(2025, 6, 30)
DOC_WINDOW = ("2025-01-01", "2025-06-30")


def _question(number: int, text: str) -> Question:
    contestant = Contestant(sql=None, answer="", time_ms=None)
    return Question(qid=f"q{number:03d}", text=text, baselines={"reference": contestant})


def _windows_reply(*items: tuple[str, str | None, str | None]) -> str:
    return json.dumps([{"qid": q, "start": s, "end": e} for q, s, e in items])


def test_question_windows_parse_overlap_and_defaults():
    questions = [_question(i, f"question {i}") for i in range(1, 6)]
    llm = FakeLLM({"question_window": _windows_reply(
        ("q001", "2024-01-01", "2024-12-31"),  # before the documents
        ("q002", "2024-12-15", "2025-01-01"),  # touches the first document day
        ("q003", "2025-06-30", "2025-09-01"),  # touches the last document day
        ("q004", None, None),  # no period named
        ("q005", "2025-05-01", "2025-04-01"),  # start after end
    )})

    result = question_windows(questions, llm, NOW, DOC_WINDOW)

    assert result == {
        "q001": {"start": "2024-01-01", "end": "2024-12-31", "in_doc_window": False},
        "q002": {"start": "2024-12-15", "end": "2025-01-01", "in_doc_window": True},
        "q003": {"start": "2025-06-30", "end": "2025-09-01", "in_doc_window": True},
        "q004": {"start": "2025-04-01", "end": "2025-06-30", "in_doc_window": True},
        "q005": {"start": "2025-04-01", "end": "2025-06-30", "in_doc_window": True},
    }


def test_question_windows_unparsable_reply_gives_default_and_batches_by_25():
    questions = [_question(i, "q") for i in range(1, 27)]
    llm = FakeLLM({"question_window": "no idea"})

    result = question_windows(questions, llm, NOW, DOC_WINDOW)

    assert len(llm.calls) == 2
    assert result["q026"] == {"start": "2025-04-01", "end": "2025-06-30", "in_doc_window": True}


def _answer() -> dict:
    text = (
        "Prices fell on drought news [doc:b1]. Exports rose sharply [doc:b2]. "
        "A tariff was announced [doc:b3]."
    )
    trace = [
        {"tool": "search_docs", "args": {}, "result": {"hits": [
            {"doc_id": "b1", "snippet": "Drought cut the wheat harvest."},
            {"doc_id": "b2", "snippet": "Exports were flat."},
        ]}},
        {"tool": "query", "args": {}, "status": "ok", "rows": [["b1", "ignored title"]]},
    ]
    return {"text": text, "tool_trace": trace}


def test_citation_validity_counts_supported_unsupported_and_no_evidence():
    llm = FakeLLM({"citation_check": per_citation(lambda block: "Drought cut" in block)})

    result = citation_validity(_answer(), llm)

    assert result == {
        "valid": 1,
        "total": 3,
        "items": [
            {"doc_id": "b1", "supported": True},
            {"doc_id": "b2", "supported": False},
            {"doc_id": "b3", "supported": False},
        ],
    }
    # One grouped call; b3 has no evidence in the trace, so only b1 and b2 are in it.
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0]
    assert prompt.count("Sentence:") == 2
    assert "[C1] Sentence: Prices fell on drought news [doc:b1]." in prompt
    first_block = prompt.split("[C2]")[0]
    assert "Drought cut the wheat harvest." in first_block
    assert "Exports were flat." not in first_block  # b2's evidence stays in b2's own block


def test_citation_validity_uses_the_preceding_claim_for_a_standalone_tag():
    llm = FakeLLM({"citation_check": per_citation(lambda block: "Prices fell." in block)})
    answer = {"text": "Prices fell. [doc:b1]", "tool_trace": _answer()["tool_trace"]}

    assert citation_validity(answer, llm)["valid"] == 1


def test_driver_checklist_counts_valid_indexes_once():
    llm = FakeLLM({"checklist": json.dumps({"covered": [0, 2, 2, 9, "x"]})})

    result = driver_checklist_score("answer", ["a", "b", "c"], llm)

    assert result == {"covered": 2, "total": 3}


def _settings(tmp_settings, tmp_path):
    return tmp_settings({"paths": {"artifacts": str(tmp_path / "artifacts"), "eval": str(tmp_path / "eval")}})


def test_quality_checks_count_roles_tags_and_write_spotcheck(tmp_settings, tmp_path):
    settings = _settings(tmp_settings, tmp_path)
    artifacts = artifacts_dir(settings)

    # 3 labelled columns: one correct, one wrong, one missing from roles.yaml.
    write_yaml(artifacts / "roles.yaml", {"tables": {"t": {"columns": {
        "a": {"role": "row_id"}, "b": {"role": "dimension"}}}}})
    write_yaml(tmp_path / "eval" / "role_labels.yaml", {"t": {"a": "row_id", "b": "location", "c": "audit"}})

    write_yaml(artifacts / "graph" / "entities_text.yaml", [
        {"id": "crop:x", "needs_review": True}, {"id": "crop:y", "needs_review": False},
        {"id": "crop:z", "needs_review": True},
    ])
    edges = [
        {"src": "d1", "rel": "TAGGED_WITH", "dst": "crop:x", "needs_review": True},
        {"src": "d1", "rel": "TAGGED_WITH", "dst": "crop:y", "needs_review": False},
        {"src": "d1", "rel": "REPORTS", "dst": "e1", "needs_review": True},
    ]
    (artifacts / "graph" / "edges.jsonl").write_text("\n".join(json.dumps(edge) for edge in edges))

    records = [{"doc_id": f"d{i:02d}", "events": [{"type": "drought", "n": i}]} for i in range(40)]
    (artifacts / "docs").mkdir()
    (artifacts / "docs" / "extractions.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    counts = quality_checks(settings)

    assert counts == {
        "role_correct": 1, "role_total": 3,
        "tag_new_entities": 3, "tag_new_flagged": 2,
        "tag_edges": 2, "tag_edges_flagged": 1,
        "spotcheck_docs": 30,
    }
    sheet = (tmp_path / "eval" / "extraction_spotcheck.md").read_text()
    assert sheet.count("\n## ") == 30
    assert quality_checks(settings) == counts
    assert (tmp_path / "eval" / "extraction_spotcheck.md").read_text() == sheet
