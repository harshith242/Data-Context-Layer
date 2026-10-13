"""Human review decisions: load, validate, merge, and track which were used."""
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class Kind(Enum):
    COLUMN = "column"
    ENTITY_TYPE = "entity_type"
    RELATION = "relation"
    EVENT_TYPE = "event_type"
    ENTITY = "entity"
    RULE = "rule"
    MOVE = "move"
    EVENT = "event"
    NAME_MATCH = "name_match"
    NEW_ENTITY = "new_entity"
    SAME_EVENT = "same_event"
    DRIVER = "driver"


class Verdict(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    FIX = "fix"


@dataclass(frozen=True)
class Decision:
    kind: Kind
    target: dict
    verdict: Verdict
    fix: dict
    by: str
    at: str
    note: str


# Required target keys per kind (spec 8.2).
_TARGET_KEYS = {
    Kind.COLUMN: ("table", "column"),
    Kind.ENTITY_TYPE: ("name",),
    Kind.RELATION: ("from", "name", "to"),
    Kind.EVENT_TYPE: ("name",),
    Kind.ENTITY: ("id",),
    Kind.RULE: ("table", "field"),
    Kind.MOVE: ("id",),
    Kind.EVENT: ("key",),
    Kind.NAME_MATCH: ("text",),
    Kind.NEW_ENTITY: ("id",),
    Kind.SAME_EVENT: ("keys",),
    Kind.DRIVER: ("move_id", "key"),
}

# Allowed `fix` keys per kind (spec 8.2). `not_entity_id` is the "wrong" payload of name_match.
_FIX_KEYS = {
    Kind.COLUMN: {"role", "logical_name", "description"},
    Kind.ENTITY_TYPE: {"name", "description"},
    Kind.RELATION: set(),
    Kind.EVENT_TYPE: {"name", "description", "merge_into"},
    Kind.ENTITY: {"canonical_name", "add_aliases", "merge_into", "parents", "part_of"},
    Kind.RULE: {"value"},
    Kind.MOVE: set(),
    Kind.EVENT: {"type", "date", "status", "affects"},
    Kind.NAME_MATCH: {"entity_id", "not_entity_id"},
    Kind.NEW_ENTITY: {"merge_into"},
    Kind.SAME_EVENT: set(),
    Kind.DRIVER: set(),
}

# Verdicts the spec marks "—" (section 8.2); a fix needs at least one fix key.
_NO_VERDICT = {
    (Kind.COLUMN, Verdict.WRONG), (Kind.RULE, Verdict.CORRECT), (Kind.RULE, Verdict.WRONG),
    (Kind.NAME_MATCH, Verdict.CORRECT),
}
_SAME_EVENT_KEYS = 2
_TOP_KEY = "decisions"


def _norm_value(kind: Kind, key: str, value):
    # A same_event pair is unordered: [k1, k2] == [k2, k1].
    if kind is Kind.SAME_EVENT and key == "keys" and isinstance(value, (list, tuple)):
        return sorted(str(item) for item in value)
    return value


def _identity(kind: Kind, target: dict) -> str:
    """Stable id of (kind, target) so later decisions can replace earlier ones."""
    parts = {key: _norm_value(kind, key, target[key]) for key in _TARGET_KEYS[kind]}
    return json.dumps([kind.value, parts], sort_keys=True, default=str)


def _target_problem(kind: Kind, target) -> str | None:
    if not isinstance(target, dict):
        return "target must be a mapping"
    for key in _TARGET_KEYS[kind]:
        if target.get(key) in (None, ""):
            return f"missing target key: {key}"
    if kind is Kind.SAME_EVENT:
        keys = target["keys"]
        if not isinstance(keys, list) or len(keys) != _SAME_EVENT_KEYS:
            return f"target keys must be a list of {_SAME_EVENT_KEYS}"
    return None


def _parse_entry(raw) -> Decision | str:
    """Return the Decision, or the reason the entry is invalid."""
    if not isinstance(raw, dict):
        return "entry must be a mapping"

    try:
        kind = Kind(raw.get("kind"))
    except ValueError:
        return f"unknown kind: {raw.get('kind')}"

    try:
        verdict = Verdict(raw.get("verdict"))
    except ValueError:
        return f"unknown verdict: {raw.get('verdict')}"

    if (kind, verdict) in _NO_VERDICT:
        return f"{kind.value} has no {verdict.value} verdict"

    problem = _target_problem(kind, raw.get("target"))
    if problem:
        return problem

    fix = raw.get("fix") or {}
    if not isinstance(fix, dict):
        return "fix must be a mapping"
    extra = sorted(set(fix) - _FIX_KEYS[kind])
    if extra:
        return f"fix key not allowed: {', '.join(map(str, extra))}"
    if verdict is Verdict.FIX and not fix:
        return "fix verdict needs a fix"

    # `at` may be an unquoted YAML date; str() gives the ISO form.
    by = str(raw.get("by") or "").strip()
    if not by:
        return "missing by"
    at = str(raw.get("at") or "").strip()
    if not at:
        return "missing at"

    return Decision(kind, dict(raw["target"]), verdict, dict(fix), by, at,
                    str(raw.get("note") or ""))


def _yaml_reason(error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None) or str(error)
    where = f" (line {mark.line + 1})" if mark else ""
    return f"YAML syntax error, file skipped: {' '.join(str(problem).split())}{where}"


class Decisions:
    def __init__(self, decisions: list[Decision], invalid: list[dict]):
        self.invalid = invalid
        self._latest = self._pick_latest(decisions)
        self._used: set[str] = set()
        self._used_since_take: set[str] = set()

    @staticmethod
    def empty() -> "Decisions":
        return Decisions([], [])

    @staticmethod
    def load(path: Path) -> "Decisions":
        """Read `path`; a missing file is empty, a broken one never raises."""
        if not path.exists():
            return Decisions.empty()

        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            return Decisions([], [{"entry": str(path), "reason": _yaml_reason(error)}])

        if data is None:
            return Decisions.empty()
        if not isinstance(data, dict) or not isinstance(data.get(_TOP_KEY), list):
            reason = f"top level must be a mapping with a '{_TOP_KEY}' list"
            return Decisions([], [{"entry": str(path), "reason": reason}])

        valid, invalid = [], []
        for raw in data[_TOP_KEY]:
            parsed = _parse_entry(raw)
            if isinstance(parsed, str):
                invalid.append({"entry": raw, "reason": parsed})
                continue
            valid.append(parsed)
        return Decisions(valid, invalid)

    @staticmethod
    def _pick_latest(decisions: list[Decision]) -> dict[str, Decision]:
        # Later `at` wins; equal `at` -> later in file wins (>= keeps the later one).
        latest: dict[str, Decision] = {}
        for decision in decisions:
            ident = _identity(decision.kind, decision.target)
            current = latest.get(ident)
            if current is None or decision.at >= current.at:
                latest[ident] = decision
        return latest

    def of(self, kind: Kind) -> list[Decision]:
        return [d for d in self._latest.values() if d.kind is kind]

    def find(self, kind: Kind, **target) -> Decision | None:
        wanted = {key: _norm_value(kind, key, value) for key, value in target.items()}
        for decision in self.of(kind):
            have = {key: _norm_value(kind, key, decision.target.get(key)) for key in wanted}
            if have == wanted:
                return decision
        return None

    def mark_used(self, decision: Decision) -> None:
        self._used.add(_identity(decision.kind, decision.target))
        self._used_since_take.add(_identity(decision.kind, decision.target))

    def take_used(self) -> list[str]:
        """Identities marked used since the last call, e.g. what one phase applied; `unmatched` is not affected."""
        taken = sorted(self._used_since_take)
        self._used_since_take = set()
        return taken

    def effective(self) -> list[tuple[str, Decision]]:
        """[(identity, decision)] for every valid decision, newest per target."""
        return list(self._latest.items())

    def unmatched(self) -> list[Decision]:
        return [d for ident, d in self._latest.items() if ident not in self._used]

    def fingerprint(self, kinds: set[Kind]) -> str:
        """Hash of the effective decisions of `kinds`; changes only when they do."""
        rows = sorted(
            (ident, d.verdict.value, d.fix)
            for ident, d in self._latest.items()
            if d.kind in kinds
        )
        return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def _raw_identity(entry) -> str | None:
    # None for entries that are not valid enough to identify; they are kept as is.
    if not isinstance(entry, dict):
        return None
    try:
        kind = Kind(entry.get("kind"))
    except ValueError:
        return None
    if _target_problem(kind, entry.get("target")):
        return None
    return _identity(kind, entry["target"])


def merge_decisions(existing: list[dict], new: list[dict]) -> list[dict]:
    """Union of both lists; per (kind, target) the newest `at` wins, ties go to the later entry."""
    entries = list(existing) + list(new)
    winner: dict[str, tuple[str, int]] = {}
    for index, entry in enumerate(entries):
        ident = _raw_identity(entry)
        if ident is None:
            continue
        rank = (str(entry.get("at") or ""), index)
        if ident not in winner or rank > winner[ident]:
            winner[ident] = rank

    merged = []
    for index, entry in enumerate(entries):
        ident = _raw_identity(entry)
        if ident is None or winner[ident][1] == index:
            merged.append(entry)
    return merged
