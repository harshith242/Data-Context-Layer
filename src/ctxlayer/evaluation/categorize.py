"""LLM categorisation of evaluation questions, with human overrides kept."""
import json
from pathlib import Path

import yaml

from ctxlayer.common.llm import LLM, extract_json
from ctxlayer.common.parallel import parallel_map
from ctxlayer.common.runlog import progress
from ctxlayer.evaluation.questions import Question

CATEGORIES = [
    "lookup", "trend_compute", "drivers", "projection", "comparison", "volatility",
    "data_trap", "adversarial", "out_of_scope", "chit_chat",
]
BEHAVIOUR_CATEGORIES = {"data_trap", "adversarial", "out_of_scope", "chit_chat"}

_SOURCE_HUMAN = "human"
_SOURCE_LLM = "llm"
_FALLBACK_CATEGORY = "lookup"
_FALLBACK_CONFIDENCE = 0.0
_MIN_CONFIDENCE = 0.0
_MAX_CONFIDENCE = 1.0

_CATEGORY_HELP = {
    "lookup": "asks for a stored value or a simple count",
    "trend_compute": "asks for a computed change, average or trend over a period",
    "drivers": "asks why something moved or what caused it",
    "projection": "asks to estimate future values",
    "comparison": "asks to compare entities, periods or sources",
    "volatility": "asks about variability, spread or stability",
    "data_trap": "looks answerable but the data cannot support a correct answer",
    "adversarial": "tries to override instructions or change or exfiltrate data",
    "out_of_scope": "asks about something the data does not cover",
    "chit_chat": "social talk with no data need",
}
_PROMPT = """\
[task: categorize]
Assign each question to exactly one category.

Categories:
{categories}

Questions (JSON; has_sql tells whether a reference query exists):
{questions}

Reply with ONLY a JSON list, one item per question:
[{{"qid": "<qid>", "category": "<category>", "confidence": <number 0 to 1>}}]"""


def categorize(
    questions: list[Question], llm: LLM, out_path: Path, batch_size: int = 25, workers: int = 1
) -> dict[str, dict]:
    """Categorise `questions`, write `out_path` (YAML) and return it as a dict.

    Entries already marked `source: human` are kept and not sent to the LLM.
    """
    known = _load_existing(out_path)
    entries = {}
    todo = []

    # Human decisions win: {"q001": {..., "source": "human"}} is never re-asked.
    for question in questions:
        existing = known.get(question.qid)
        if existing and existing.get("source") == _SOURCE_HUMAN:
            entries[question.qid] = existing
            continue
        todo.append(question)

    batches = [todo[start : start + batch_size] for start in range(0, len(todo), batch_size)]
    for result in parallel_map(
        lambda batch: _categorize_batch(batch, llm), batches, workers,
        lambda done, total: progress("eval_prepare.categorize_batches", done, total),
    ):
        entries.update(result)

    ordered = {question.qid: entries[question.qid] for question in questions}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(ordered, sort_keys=False))

    return ordered


def _load_existing(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _categorize_batch(batch: list[Question], llm: LLM) -> dict[str, dict]:
    # Only id, text and has_sql leave the process: never SQL or answers.
    payload = [
        {"qid": q.qid, "question": q.text, "has_sql": any(c.sql is not None for c in q.baselines.values())}
        for q in batch
    ]
    categories = "\n".join(f"- {name}: {_CATEGORY_HELP[name]}" for name in CATEGORIES)
    prompt = _PROMPT.format(categories=categories, questions=json.dumps(payload, indent=1))

    verdicts = _verdicts_by_qid(llm.complete(prompt).text)

    return {
        item["qid"]: _entry(question, item["has_sql"], verdicts.get(item["qid"]))
        for question, item in zip(batch, payload)
    }


def _verdicts_by_qid(reply_text: str) -> dict[str, dict]:
    """{"q001": {...}} from the reply list; empty when the reply is not a JSON list."""
    items = extract_json(reply_text)
    if not isinstance(items, list):
        return {}
    return {
        item["qid"]: item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("qid"), str)
    }


def _entry(question: Question, has_sql: bool, verdict: dict | None) -> dict:
    category = verdict.get("category") if verdict else None
    confidence = verdict.get("confidence") if verdict else None
    valid = category in CATEGORIES and _is_probability(confidence)

    # Missing or invalid verdict -> safest default, flagged for a human.
    return {
        "question": question.text,
        "category": category if valid else _FALLBACK_CATEGORY,
        "confidence": confidence if valid else _FALLBACK_CONFIDENCE,
        "has_sql": has_sql,
        "needs_review": not valid,
        "source": _SOURCE_LLM,
    }


def _is_probability(value) -> bool:
    # bool is an int subclass: True would pass as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return _MIN_CONFIDENCE <= value <= _MAX_CONFIDENCE
