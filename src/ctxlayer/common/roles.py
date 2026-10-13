"""Column role detection: profile heuristics, then one LLM call per table, validated."""
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from ctxlayer.common.config import Settings
from ctxlayer.common.context_files import ContextFiles, Step
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.llm import LLM, LLMError, extract_json
from ctxlayer.common.review import flag

ROLES = [
    "row_id", "entity_id", "entity_label", "hierarchy_level", "dimension", "location",
    "time_business", "time_other", "period_key", "measure", "measure_band", "scenario",
    "currency", "unit", "frequency", "doc_id", "text_title", "text_summary", "text_body",
    "text_part", "doc_tag", "audit",
]

_ROLE_MEANINGS = {
    "row_id": "surrogate key",
    "entity_id": "identifier of the main measured entity",
    "entity_label": "name of the main measured entity",
    "hierarchy_level": "a level in a dimension hierarchy (parent levels show up as functional dependencies)",
    "dimension": "other categorical attribute",
    "location": "geographic dimension",
    "time_business": "the date to report and filter by (at most one per table)",
    "time_other": "any other date or timestamp; give its meaning",
    "period_key": "integer period such as YYYYMM",
    "measure": "numeric value; give additive (true if summing rows is meaningful)",
    "measure_band": "one of several measures forming a low / mid / high band; give band",
    "scenario": "column telling observed values from projected ones",
    "currency": "currency of the measures",
    "unit": "unit of the measures",
    "frequency": "sampling frequency of the rows",
    "doc_id": "identifier of the document a row belongs to (rows repeat per document)",
    "text_title": "document title",
    "text_summary": "document summary",
    "text_body": "document body in one column",
    "text_part": "one part of a document body split over columns; give part_order (1 = first)",
    "doc_tag": "metadata tag on a document (a thing the document is about)",
    "audit": "record keeping, ignored",
}

_INFERRED = "inferred"
_HUMAN = "human"
_FROM_CONTEXT = "context.md"
_STRONG_HEURISTIC = 0.9
_FALLBACK_MAX_CONFIDENCE = 0.5
_TEXT_MIN_AVG_LEN = 200
_FLOAT_TYPES = ("DOUBLE", "FLOAT", "REAL", "DECIMAL")
_MEASURE_ROLES = ("measure", "measure_band")
_SCENARIO_LABELS = ("actual", "projected")

_PROMPT = """[task: roles]
Assign one role to every column of one database table.

Table: {table}
Rows: {rows}

Roles (use exactly these names):
{roles}

Context from the data owner (may be empty):
{context}

Columns, one JSON object per line. "heuristic" is a rule-based guess and may be wrong:
{columns}

Functional dependencies [a, b]: every a maps to one b, so b is a parent level of a:
{fds}

Repeat keys: columns whose values repeat over rows, with the columns that vary within one value:
{repeat_keys}

Reply with one JSON object and nothing else:
{{
  "kind": "measures" | "text" | "other",
  "primary_measure": "<column of role measure or measure_band, measures tables only>",
  "scenario_values": {{"<value of the scenario column>": {scenario_labels}}},
  "columns": {{
    "<column>": {{
      "role": "<role>",
      "logical_name": "<unique snake_case name>",
      "description": "<one short sentence>",
      "confidence": <0..1>,
      "band": "low" | "mid" | "high",
      "part_order": <int>,
      "additive": <true | false>,
      "meaning": "<what a time_other date means>",
      "from_context": <true if the context text decided this role>
    }}
  }}
}}
Include every column. Keep band, part_order, additive and meaning only where the role calls for them.
Omit primary_measure and scenario_values when the table has no measures or scenario."""


class _ColumnReply(BaseModel):
    role: Literal[*ROLES]
    logical_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    confidence: float = Field(ge=0, le=1)
    band: Literal["low", "mid", "high"] | None = None
    part_order: int | None = None
    additive: bool | None = None
    meaning: str | None = None
    from_context: bool = False


class _TableReply(BaseModel):
    kind: Literal["measures", "text", "other"]
    primary_measure: str | None = None
    scenario_values: dict[str, str] = {}
    columns: dict[str, Any]

    @field_validator("scenario_values")
    @classmethod
    def _known_labels(cls, values: dict[str, str]) -> dict[str, str]:
        # Drop unknown labels (e.g. "estimate") instead of rejecting the whole table reply.
        return {value: label for value, label in values.items() if label in _SCENARIO_LABELS}


def heuristic_roles(table_profile: dict) -> dict[str, dict]:
    """{column: {role, confidence, reason}} from types, cardinality and patterns only."""
    columns = table_profile["columns"]
    rows = table_profile["rows"]

    roles = {name: _column_guess(stats, rows) for name, stats in columns.items()}
    roles.update(_document_guess(table_profile))
    return roles


def _guess(role: str, confidence: float, reason: str) -> dict:
    return {"role": role, "confidence": confidence, "reason": reason}


def _is_text(stats: dict) -> bool:
    return stats["type"] == "VARCHAR" and (stats["avg_len"] or 0) >= _TEXT_MIN_AVG_LEN


def _column_guess(stats: dict, rows: int) -> dict:
    sql_type = stats["type"]

    if stats["is_integer"] and stats["distinct"] == rows and stats["null_pct"] == 0:
        return _guess("row_id", 0.9, "integer, unique per row")
    if stats["looks_yyyymm"]:
        return _guess("period_key", 0.9, "values look like YYYYMM")
    if sql_type == "DATE":
        return _guess("time_business", 0.6, "date column")
    if sql_type.startswith("TIMESTAMP"):
        return _guess("time_business", 0.5, "timestamp column")
    if sql_type.startswith(_FLOAT_TYPES):
        return _guess("measure", 0.7, "floating-point column")
    if _is_text(stats):
        return _guess("text_body", 0.5, "long text")
    if sql_type == "VARCHAR" and "top_values" in stats:
        return _guess("dimension", 0.5, "low-cardinality text")
    return _guess("dimension", 0.3, "no rule matched")


def _document_guess(table_profile: dict) -> dict[str, dict]:
    """doc_id and doc_tag columns of a table with long text and a repeating short-text key."""
    columns = table_profile["columns"]
    if not any(_is_text(stats) for stats in columns.values()):
        return {}

    keys = [
        key for key in table_profile["repeat_keys"]
        if columns[key["column"]]["type"] == "VARCHAR" and not _is_text(columns[key["column"]])
    ]
    if not keys:
        return {}

    # Most distinct values, then shortest: a reference like "REF-0001" beats a headline.
    best = max(keys, key=lambda key: (columns[key["column"]]["distinct"], -columns[key["column"]]["avg_len"]))
    guesses = {best["column"]: _guess("doc_id", 0.6, "short text repeating over rows beside long text")}

    for name in best["varying_columns"]:
        stats = columns[name]
        if stats["type"] == "VARCHAR" and "top_values" in stats and not _is_text(stats):
            guesses[name] = _guess("doc_tag", 0.6, f"low-cardinality text varying within {best['column']}")
    return guesses


def detect_roles(profile: dict, llm: LLM, context: ContextFiles, settings: Settings) -> dict:
    """roles.yaml content: {"tables": {table: {kind, columns: {column: {...}}, ...}}}."""
    threshold = settings.get("review.confidence_threshold")

    tables = {}
    for table, table_profile in profile["tables"].items():
        tables[table] = _table_roles(table, table_profile, llm, context.for_step(Step.ROLES, table), threshold)
    return {"tables": tables}


def apply_column_decisions(roles: dict, decisions: Decisions) -> dict:
    """Apply `column` decisions to freshly detected roles; reviewed columns get source human and confidence 1.0."""
    for decision in decisions.of(Kind.COLUMN):
        columns = roles["tables"].get(decision.target["table"], {}).get("columns", {})
        item = columns.get(decision.target["column"])
        if item is None or decision.verdict is Verdict.WRONG:
            continue
        if decision.verdict is Verdict.FIX and not _fix_column(columns, item, decision.fix):
            continue

        item.update(confidence=1.0, needs_review=False, source=_HUMAN)
        decisions.mark_used(decision)
    return roles


def _fix_column(columns: dict, item: dict, fix: dict) -> bool:
    """Set the fixed fields on `item`; False (nothing changed) for a role that does not exist."""
    if fix.get("role", item["role"]) not in ROLES:
        return False

    # A logical name another column already has would break the one-name-per-column rule: keep the old one.
    taken = {other["logical_name"] for other in columns.values() if other is not item}
    if fix.get("logical_name") in taken:
        fix = {key: value for key, value in fix.items() if key != "logical_name"}
    item.update(fix)
    return True


def _table_roles(table: str, table_profile: dict, llm: LLM, context_text: str, threshold: float) -> dict:
    heuristic = heuristic_roles(table_profile)
    reply = _ask(table, table_profile, heuristic, llm, context_text)
    replied = reply.columns if reply else {}

    used_names: set[str] = set()
    columns = {}
    for name in table_profile["columns"]:
        columns[name] = _assign(name, heuristic[name], replied.get(name), context_text, used_names, threshold)
    _keep_one_business_time(columns)

    kind = reply.kind if reply else _guess_kind(columns)
    result = {"kind": kind}
    if kind == "measures":
        primary = _primary_measure(reply, columns)
        if primary:
            result["primary_measure"] = primary
        result["scenario_values"] = reply.scenario_values if reply else {}
        _flag_unlabelled_scenario(columns, table_profile, result["scenario_values"])
    result["columns"] = columns
    return result


def _flag_unlabelled_scenario(columns: dict, table_profile: dict, labels: dict) -> None:
    """A scenario column with any frequent value left unlabelled needs a human (metrics would drop those rows)."""
    for name, item in columns.items():
        if item["role"] != "scenario":
            continue
        values = {str(value) for value, _ in table_profile["columns"][name].get("top_values", [])}
        if not labels or not values <= set(labels):
            item["needs_review"] = True


def _ask(table: str, table_profile: dict, heuristic: dict, llm: LLM, context_text: str) -> _TableReply | None:
    """The validated table-level reply, or None when the LLM fails or answers badly."""
    column_lines = [
        json.dumps({"name": name, **stats, "heuristic": _public_guess(heuristic[name])}, default=str)
        for name, stats in table_profile["columns"].items()
    ]
    prompt = _PROMPT.format(
        table=table,
        rows=table_profile["rows"],
        scenario_labels=" | ".join(f'"{label}"' for label in _SCENARIO_LABELS),
        roles="\n".join(f"- {role}: {_ROLE_MEANINGS[role]}" for role in ROLES),
        context=context_text or "(none)",
        columns="\n".join(column_lines),
        fds=json.dumps(table_profile["fds"]),
        repeat_keys=json.dumps(table_profile["repeat_keys"]),
    )

    try:
        text = llm.complete(prompt).text
    except LLMError:
        return None

    data = extract_json(text)
    if not isinstance(data, dict):
        return None
    try:
        return _TableReply.model_validate(data)
    except ValidationError:
        return None


def _public_guess(guess: dict) -> dict:
    return {"role": guess["role"], "confidence": guess["confidence"]}


def _assign(name: str, guess: dict, raw: Any, context_text: str, used_names: set, threshold: float) -> dict:
    """The LLM's assignment for a column if valid, else the heuristic one marked for review."""
    try:
        parsed = _ColumnReply.model_validate(raw)
    except ValidationError:
        return _fallback(name, guess, used_names, threshold)
    if parsed.logical_name in used_names:
        return _fallback(name, guess, used_names, threshold)

    used_names.add(parsed.logical_name)
    item = parsed.model_dump(exclude={"from_context"}, exclude_none=True)
    item["source"] = _FROM_CONTEXT if parsed.from_context and context_text else _INFERRED
    flag(item, threshold)

    # The LLM overruled a strong rule: keep its answer, ask a human to confirm.
    if parsed.role != guess["role"] and guess["confidence"] >= _STRONG_HEURISTIC:
        item["needs_review"] = True
    return item


def _fallback(name: str, guess: dict, used_names: set, threshold: float) -> dict:
    logical = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "column"
    while logical in used_names:
        logical += "_2"
    used_names.add(logical)

    item = {
        "role": guess["role"],
        "logical_name": logical,
        "description": guess["reason"],
        "confidence": min(guess["confidence"], _FALLBACK_MAX_CONFIDENCE),
        "source": _INFERRED,
    }
    flag(item, threshold)
    item["needs_review"] = True
    return item


def _keep_one_business_time(columns: dict) -> None:
    """Two time_business columns: the more confident one stays, the rest become time_other."""
    timed = [name for name, item in columns.items() if item["role"] == "time_business"]
    if len(timed) < 2:
        return

    best = max(timed, key=lambda name: columns[name]["confidence"])
    for name in timed:
        if name == best:
            continue
        columns[name]["role"] = "time_other"
        columns[name]["needs_review"] = True


def _guess_kind(columns: dict) -> str:
    roles = {item["role"] for item in columns.values()}
    if roles & {"doc_id", "text_title", "text_summary", "text_body", "text_part"}:
        return "text"
    if roles & set(_MEASURE_ROLES):
        return "measures"
    return "other"


def _primary_measure(reply: _TableReply | None, columns: dict) -> str | None:
    """The LLM's choice if it is a measure column, else the mid band or the first measure."""
    measures = [name for name, item in columns.items() if item["role"] in _MEASURE_ROLES]
    if reply and reply.primary_measure in measures:
        return reply.primary_measure

    for name in measures:
        if columns[name].get("band") == "mid":
            return name
    return measures[0] if measures else None
