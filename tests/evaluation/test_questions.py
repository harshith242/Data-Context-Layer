"""Question CSV loader."""
import pandas as pd
import pytest

from ctxlayer.evaluation.questions import load_questions, reference_baseline


def _baselines(tmp_settings) -> list[dict]:
    return tmp_settings().get("eval.baselines")


def test_load_questions_maps_all_columns_for_a_query_row(domain_a, tmp_settings):
    frame = pd.read_csv(domain_a.questions_csv)

    questions = load_questions(domain_a.questions_csv, "question", _baselines(tmp_settings))
    first = questions[0]

    assert len(questions) == 8
    assert first.qid == "q001"
    assert first.text == frame.loc[0, "question"]
    assert list(first.baselines) == ["reference"]
    assert first.baselines["reference"].sql == frame.loc[0, "reference_sql"]
    assert first.baselines["reference"].answer == frame.loc[0, "reference_answer"]
    assert first.baselines["reference"].time_ms == int(frame.loc[0, "reference_time_ms"])


def test_load_questions_turns_missing_sql_into_none_and_keeps_time_int(domain_a, tmp_settings):
    questions = load_questions(domain_a.questions_csv, "question", _baselines(tmp_settings))
    no_sql = questions[3]

    assert no_sql.qid == "q004"
    assert no_sql.baselines["reference"].sql is None
    assert isinstance(no_sql.baselines["reference"].time_ms, int)
    assert [q.qid for q in questions if q.baselines["reference"].sql is None] == ["q004", "q005", "q006", "q007"]


def test_load_questions_follows_the_configured_baselines(tmp_path):
    csv_path = tmp_path / "q.csv"
    pd.DataFrame(
        {"Q": ["Q?"], "a_sql": ["SELECT 1"], "a_ms": [10], "a_ans": ["x"], "b_sql": ["SELECT 2"], "b_ans": ["y"]}
    ).to_csv(csv_path, index=False)
    baselines = [
        {"name": "first", "sql_column": "a_sql", "time_ms_column": "a_ms", "answer_column": "a_ans"},
        {"name": "second", "sql_column": "b_sql", "answer_column": "b_ans"},
    ]

    (question,) = load_questions(csv_path, "Q", baselines)

    first, second = question.baselines["first"], question.baselines["second"]
    assert (first.sql, first.time_ms, first.answer) == ("SELECT 1", 10, "x")
    assert (second.sql, second.time_ms, second.answer) == ("SELECT 2", None, "y")


def test_load_questions_without_baselines_keeps_only_the_question(tmp_path):
    csv_path = tmp_path / "q.csv"
    pd.DataFrame({"question": ["Q?"]}).to_csv(csv_path, index=False)

    (question,) = load_questions(csv_path, "question", [])

    assert question.text == "Q?" and question.baselines == {}


@pytest.mark.parametrize(
    "baselines, expected",
    [
        ([], None),
        ([{"name": "a", "answer_column": "x"}], None),
        ([{"name": "a", "answer_column": "x"}, {"name": "b", "sql_column": "s"}, {"name": "c", "sql_column": "t"}], "b"),
    ],
)
def test_reference_baseline_is_the_first_one_with_sql(baselines, expected):
    assert reference_baseline(baselines) == expected
