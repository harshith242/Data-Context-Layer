"""Evaluation question import from the questions CSV."""
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ctxlayer.common.llm import LLM, extract_json
from ctxlayer.common.parallel import parallel_map
from ctxlayer.common.runlog import progress

_WINDOW_BATCH = 25
_DEFAULT_WINDOW_DAYS = 90
_WINDOW_PROMPT = """\
[task: question_window]
For each question give the period of time it refers to, as ISO dates (YYYY-MM-DD).
Use null for start and end when the question names no period. Today is {now}.

Questions (JSON):
{questions}

Reply with ONLY a JSON list, one item per question:
[{{"qid": "<qid>", "start": "<date or null>", "end": "<date or null>"}}]"""


@dataclass
class Contestant:
    sql: str | None
    answer: str
    time_ms: int | None


@dataclass
class Question:
    qid: str
    text: str
    baselines: dict[str, Contestant]  # by baseline name, in settings order


def reference_baseline(baselines: list[dict]) -> str | None:
    """Name of the first baseline with an SQL column: the reference for exec and correctness scoring."""
    for baseline in baselines:
        if baseline.get("sql_column"):
            return baseline["name"]
    return None


def load_questions(csv_path: Path, question_column: str, baselines: list[dict]) -> list[Question]:
    """One Question per CSV row; each baseline dict names its columns (name, sql_column, answer_column, time_ms_column)."""
    frame = pd.read_csv(csv_path)
    questions = []

    for index, row in frame.iterrows():
        questions.append(
            Question(
                qid=f"q{index + 1:03d}",
                text=row[question_column],
                baselines={baseline["name"]: _contestant(row, baseline) for baseline in baselines},
            )
        )

    return questions


def _contestant(row: pd.Series, baseline: dict) -> Contestant:
    sql = _cell(row, baseline.get("sql_column"))
    answer = _cell(row, baseline.get("answer_column"))
    time_ms = _cell(row, baseline.get("time_ms_column"))

    # Unset column or empty CSV cell (NaN): no SQL -> None, no time -> None, no answer -> "".
    return Contestant(
        sql=None if pd.isna(sql) else sql,
        answer="" if pd.isna(answer) else answer,
        time_ms=None if pd.isna(time_ms) else int(time_ms),
    )


def _cell(row: pd.Series, column: str | None):
    return None if column is None else row[column]


def question_windows(
    questions: list[Question], llm: LLM, now: date, doc_window: tuple[str, str], workers: int = 1
) -> dict[str, dict]:
    """{qid: {start, end, in_doc_window}}; no usable period -> the 90 days before `now`."""
    windows = {}

    # Batches of questions asked `workers` at a time.
    batches = [questions[first : first + _WINDOW_BATCH] for first in range(0, len(questions), _WINDOW_BATCH)]
    all_replies = parallel_map(lambda batch: _window_replies(batch, llm, now), batches, workers)

    for batch, replies in zip(batches, all_replies):
        for question in batch:
            start, end = _parse_window(replies.get(question.qid), now)
            windows[question.qid] = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "in_doc_window": _overlaps(start, end, doc_window),
            }
        progress("eval_prepare.question_windows", len(windows), len(questions))

    return windows


def _window_replies(batch: list[Question], llm: LLM, now: date) -> dict[str, dict]:
    payload = [{"qid": q.qid, "question": q.text} for q in batch]
    prompt = _WINDOW_PROMPT.format(now=now.isoformat(), questions=json.dumps(payload, indent=1))

    items = extract_json(llm.complete(prompt).text)
    if not isinstance(items, list):
        return {}
    return {item["qid"]: item for item in items if isinstance(item, dict) and "qid" in item}


def _parse_window(reply: dict | None, now: date) -> tuple[date, date]:
    default = (now - timedelta(days=_DEFAULT_WINDOW_DAYS), now)
    if not reply:
        return default

    # null, non-string or malformed dates, or start after end -> default window.
    try:
        start = date.fromisoformat(reply["start"])
        end = date.fromisoformat(reply["end"])
    except (KeyError, TypeError, ValueError):
        return default
    return (start, end) if start <= end else default


def _overlaps(start: date, end: date, doc_window: tuple[str, str]) -> bool:
    doc_start, doc_end = (date.fromisoformat(value) for value in doc_window)
    return start <= doc_end and end >= doc_start
