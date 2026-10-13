"""Drafts semantic.yaml from roles, profile, source data and context.md; the LLM adds grains, conversions, synonyms."""
import json
import numbers

from ctxlayer.common.context_files import ContextFiles, Step
from ctxlayer.common.decisions import Decisions, Kind
from ctxlayer.common.db import quote_ident
from ctxlayer.common.llm import LLM, LLMError, extract_json

_INFERRED = "inferred"
_FROM_CONTEXT = "context.md"
_HUMAN = "human"
_SYNONYMS_PREFIX = "synonyms."
_CONVERSIONS = "conversions"
_ACTUAL = "actual"
_AUDIT = "audit"
_MEASURE_ROLES = ("measure", "measure_band")
_TABLE_KINDS = ("measures", "text")
_AGGREGATION = "avg"
_GRAINS = ["day", "week", "month", "quarter", "year"]
_MAX_FREQUENCY_VALUES = 50

_GRAINS_PROMPT = """[task: grains]
Map each sampling-frequency value of a table to the time grain of its rows.

Table: {table}
Grains (use exactly these names): {grains}
Frequency values: {values}

Context from a person (may be empty):
{context}

Reply with one JSON object and nothing else: {{"<frequency value>": "<grain>"}}"""

_CONVERSIONS_PROMPT = """[task: conversions]
List the conversion factors between currencies or units that the text states explicitly.
Never invent a factor. Reply [] when there is none.

Columns (logical names): {columns}

Text from the data owner:
{context}

Reply with one JSON list and nothing else:
[{{"column": "<column>", "from": "<value>", "to": "<value>", "factor": <number: multiply "from" by it to get "to">}}]"""

_SYNONYMS_PROMPT = """[task: synonyms]
Suggest the other names people use for the columns of one table.

Table: {table}
Context from the data owner (may be empty):
{context}

Columns, "<name>: <description>":
{columns}

Reply with one JSON object and nothing else: {{"<column>": ["<synonym>", ...]}}
Leave out columns without a useful synonym."""


def draft_semantic(roles: dict, profile: dict, con, llm: LLM, context: ContextFiles) -> dict:
    """semantic.yaml content; `con` reads the source database."""
    tables = {}
    for table, table_roles in roles["tables"].items():
        if table_roles["kind"] not in _TABLE_KINDS:
            continue
        notes = context.for_step(Step.SEMANTIC, table)
        tables[table] = _draft_table(table, table_roles, profile["tables"][table], con, llm, notes)

    return {
        "now": {"value": _now(roles, con), "source": _INFERRED},
        "tables": tables,
        "conversions": _conversions(tables, llm, context),
        "synonyms": _synonyms(tables, llm, context),
    }


def apply_rule_decisions(semantic: dict, decisions: Decisions) -> dict:
    """Set the fields a `rule` decision names to its value; the fields are listed under `human_rules`.

    Fields: primary_measure, scenario.values, denominations, time.grains (of the decision's table) and
    synonyms.<name>, conversions (global: the table is not used). A value of the wrong shape is not applied.
    """
    human = []
    for decision in decisions.of(Kind.RULE):
        value = decision.fix.get("value")
        if value is None or not _set_rule(semantic, decision.target, value):
            continue

        decisions.mark_used(decision)
        human.append({"table": decision.target["table"], "field": decision.target["field"]})
    if human:
        semantic["human_rules"] = human
    return semantic


def _set_rule(semantic: dict, target: dict, value) -> bool:
    field = target["field"]
    tables = semantic["tables"]
    logical = sorted({column["logical"] for spec in tables.values() for column in spec["columns"]})

    if field.startswith(_SYNONYMS_PREFIX):
        name = field[len(_SYNONYMS_PREFIX):]
        if name not in logical or not _strings(value):
            return False
        semantic["synonyms"][name] = value
        return True

    if field == _CONVERSIONS:
        if not isinstance(value, list) or not all(_valid_conversion(entry, logical) for entry in value):
            return False
        semantic["conversions"] = [{**entry, "source": _HUMAN} for entry in value]
        return True

    spec = tables.get(target["table"])
    if spec is None:
        return False
    return _set_table_rule(spec, field, value)


def _set_table_rule(spec: dict, field: str, value) -> bool:
    columns = {column["logical"] for column in spec["columns"]}
    if field == "primary_measure":
        if value not in {measure["logical"] for measure in spec.get("measures", [])}:
            return False
        spec[field] = value
    elif field == "denominations":
        if not _strings(value) or not set(value) <= columns:
            return False
        spec[field] = value
    elif field == "time.grains":
        if "time" not in spec or not _strings(value) or not value or not set(value) <= set(_GRAINS):
            return False
        spec["time"]["grains"] = value
    elif field == "scenario.values":
        if "scenario" not in spec or not isinstance(value, dict):
            return False
        spec["scenario"].update(values=value, source=_HUMAN)
    else:
        return False
    return True


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _draft_table(table: str, table_roles: dict, table_profile: dict, con, llm: LLM, notes: str) -> dict:
    columns = {name: item for name, item in table_roles["columns"].items() if item["role"] != _AUDIT}
    draft = {
        "view": f"sem_{_slug(table)}",
        "kind": table_roles["kind"],
        "columns": [_column(name, item) for name, item in columns.items()],
    }

    _put(draft, "entity", _entity(columns))
    _put(draft, "hierarchy", _hierarchy(columns, table_profile["fds"]))
    _put(draft, "time", _time(table, columns, con, llm, notes))

    if table_roles["kind"] == "measures":
        _put(draft, "measures", _measures(columns))
        _put(draft, "primary_measure", _logical_of(columns, table_roles.get("primary_measure")))
        draft["denominations"] = _denominations(columns)  # always present; [] means a single denomination
        _put(draft, "scenario", _scenario(columns, table_roles))
    _put(draft, "frequency", _logical(columns, "frequency"))
    return draft


def _put(draft: dict, key: str, value) -> None:
    """Set `key` unless `value` is empty, so absent roles leave no key."""
    if value:
        draft[key] = value


def _slug(table: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in table.lower())


def _column(name: str, item: dict) -> dict:
    column = {
        "physical": name,
        "logical": item["logical_name"],
        "role": item["role"],
        "description": item.get("description", ""),
        "source": item["source"],
        "confidence": item["confidence"],
    }
    if item.get("needs_review"):
        column["needs_review"] = True
    return column


def _physical(columns: dict, role: str) -> str | None:
    return next((name for name, item in columns.items() if item["role"] == role), None)


def _logical(columns: dict, role: str) -> str | None:
    name = _physical(columns, role)
    return columns[name]["logical_name"] if name else None


def _logical_of(columns: dict, physical: str | None) -> str | None:
    return columns[physical]["logical_name"] if physical in columns else None


def _entity(columns: dict) -> dict:
    found = {"id": _logical(columns, "entity_id"), "label": _logical(columns, "entity_label")}
    return {key: logical for key, logical in found.items() if logical}


def _denominations(columns: dict) -> list[str]:
    found = [_logical(columns, "currency"), _logical(columns, "unit")]
    return [logical for logical in found if logical]


def _hierarchy(columns: dict, fds: list[list[str]]) -> list[str]:
    """Hierarchy levels, child first. [level_a, level_b] in fds means level_b is the parent of level_a."""
    levels = [name for name, item in columns.items() if item["role"] == "hierarchy_level"]

    # A level with more parent levels is lower: level_a (1 parent) before level_b (0).
    def parent_count(level: str) -> int:
        return sum(1 for child, parent in fds if child == level and parent in levels)

    ordered = sorted(levels, key=parent_count, reverse=True)
    return [columns[name]["logical_name"] for name in ordered]


def _measures(columns: dict) -> list[dict]:
    measures = []
    for item in columns.values():
        if item["role"] not in _MEASURE_ROLES:
            continue

        # Unknown additivity: assume summing is not meaningful.
        measure = {"logical": item["logical_name"], "band": item.get("band"), "additive": item.get("additive", False)}
        measures.append({**measure, "aggregation": _AGGREGATION})
    return measures


def _scenario(columns: dict, table_roles: dict) -> dict | None:
    physical = _physical(columns, "scenario")
    if not physical:
        return None
    return {
        "column": columns[physical]["logical_name"], "values": table_roles.get("scenario_values", {}),
        "default": _ACTUAL, "source": columns[physical].get("source", _INFERRED),
    }


def _time(table: str, columns: dict, con, llm: LLM, notes: str) -> dict | None:
    business = _physical(columns, "time_business")
    if not business:
        return None

    low, high = con.execute(
        f"SELECT CAST(min({quote_ident(business)}) AS DATE), CAST(max({quote_ident(business)}) AS DATE) FROM {quote_ident(table)}"
    ).fetchone()
    time = {"business": columns[business]["logical_name"]}
    _put(time, "period_key", _logical(columns, "period_key"))
    time["range"] = [low.isoformat() if low else None, high.isoformat() if high else None]

    # Unknown native grain (no frequency column or unusable reply): every grain is allowed; date_trunc copes.
    frequency = _physical(columns, "frequency")
    time["grains"] = (_grains(table, frequency, con, llm, notes) if frequency else []) or list(_GRAINS)
    return time


def _grains(table: str, frequency: str, con, llm: LLM, notes: str) -> list[str]:
    """Grains the table supports: the finest one its frequency values map to, and all coarser ones."""
    values = [
        value for (value,) in con.execute(
            f"SELECT DISTINCT {quote_ident(frequency)} FROM {quote_ident(table)} WHERE {quote_ident(frequency)} IS NOT NULL"
            f" ORDER BY 1 LIMIT {_MAX_FREQUENCY_VALUES}"
        ).fetchall()
    ]
    prompt = _GRAINS_PROMPT.format(
        table=table, grains=", ".join(_GRAINS), values=json.dumps(values, default=str), context=notes,
    )
    reply = _ask(llm, prompt)
    if not isinstance(reply, dict):
        return []

    # {"every 7 days": "week", "every 4 weeks": "month"} -> position of "week" -> week..year
    positions = [_GRAINS.index(grain) for grain in reply.values() if grain in _GRAINS]
    if not positions:
        return []
    return _GRAINS[min(positions):]


def _now(roles: dict, con) -> str | None:
    """Latest business date among rows labelled actual, over all measures tables."""
    latest = []
    for table, table_roles in roles["tables"].items():
        if table_roles["kind"] != "measures":
            continue

        columns = table_roles["columns"]
        business = _physical(columns, "time_business")
        if not business:
            continue

        sql = f"SELECT CAST(max({quote_ident(business)}) AS DATE) FROM {quote_ident(table)}"
        scenario = _physical(columns, "scenario")
        if scenario:
            actual = [value for value, label in table_roles.get("scenario_values", {}).items() if label == _ACTUAL]
            if not actual:
                continue
            literals = ", ".join("'" + value.replace("'", "''") + "'" for value in actual)
            sql += f" WHERE {quote_ident(scenario)} IN ({literals})"

        value = con.execute(sql).fetchone()[0]
        if value:
            latest.append(value)
    return max(latest).isoformat() if latest else None


def _conversions(tables: dict, llm: LLM, context: ContextFiles) -> list[dict]:
    """Factors stated in context.md; none without context text."""
    text = context.for_step(Step.SEMANTIC)
    if not text:
        return []

    logical = sorted({column["logical"] for table in tables.values() for column in table["columns"]})
    reply = _ask(llm, _CONVERSIONS_PROMPT.format(columns=", ".join(logical), context=text))
    if not isinstance(reply, list):
        return []

    conversions = []
    for entry in reply:
        if not _valid_conversion(entry, logical):
            continue
        conversions.append({
            "column": entry["column"], "from": str(entry["from"]), "to": str(entry["to"]),
            "factor": entry["factor"], "source": _FROM_CONTEXT,
        })
    return conversions


def _valid_conversion(entry, logical: list[str]) -> bool:
    if not isinstance(entry, dict) or entry.get("column") not in logical:
        return False
    if "from" not in entry or "to" not in entry:
        return False

    factor = entry.get("factor")
    return isinstance(factor, numbers.Real) and not isinstance(factor, bool) and factor > 0


def _synonyms(tables: dict, llm: LLM, context: ContextFiles) -> dict[str, list[str]]:
    """{logical: [synonym]} from one LLM call per measures table."""
    synonyms: dict[str, list[str]] = {}
    for table, draft in tables.items():
        if draft["kind"] != "measures":
            continue

        described = {column["logical"]: column["description"] for column in draft["columns"]}
        prompt = _SYNONYMS_PROMPT.format(
            table=table,
            context=context.for_step(Step.SEMANTIC, table) or "(none)",
            columns="\n".join(f"{name}: {description}" for name, description in described.items()),
        )
        reply = _ask(llm, prompt)
        if not isinstance(reply, dict):
            continue

        for name, words in reply.items():
            if name not in described or not isinstance(words, list):
                continue
            known = synonyms.get(name, [])
            for word in words:
                if isinstance(word, str) and word != name and word not in known:
                    known.append(word)
            if known:
                synonyms[name] = known
    return synonyms


def _ask(llm: LLM, prompt: str):
    """Parsed JSON of the reply; None when the LLM fails or answers without JSON."""
    try:
        return extract_json(llm.complete(prompt).text)
    except LLMError:
        return None
