"""Entity resolver: map a free-text mention to catalog entities (spec section 6.3)."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from ctxlayer.common.catalog import Entity
from ctxlayer.common.config import Settings
from ctxlayer.common.decisions import Decisions, Kind
from ctxlayer.common.embeddings import Embedder, cosine_top_k
from ctxlayer.common.llm import LLM, LLMError, extract_json

_DEFAULT_K = 5
_TASK_JUDGE = "resolve_judge"
_TASK_JUDGE_BATCH = "resolve_judge_batch"
_BATCH_ATTEMPTS = 2  # one retry of a batch whose reply is not usable JSON
_VERDICT_FILE = "resolve_judge.jsonl"
_EXACT_SCORE = 1.0
_PERCENT = 100
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_CONTEXT_BLOCK = "Context from a person (may be empty):\n{context}\n"


class Status(str, Enum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


class Method(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    EMBEDDING = "embedding"
    HUMAN = "human"  # a name_match decision


class Cascade(Enum):
    STRICT = "strict"  # exact + alias only
    FULL = "full"  # exact -> alias -> fuzzy -> embedding + LLM judge


@dataclass
class Match:
    id: str
    name: str
    type: str
    score: float
    method: Method
    needs_review: bool


@dataclass
class ResolveResult:
    status: Status
    matches: list[Match]
    suggestions: list[str]


class Resolver:
    def __init__(
        self,
        entities: list[Entity],
        embedder: Embedder,
        llm: LLM | None,
        settings: Settings,
        cascade: Cascade = Cascade.FULL,
        cache_path: Path | None = None,
        context: str = "",
        overrides: dict[str, str] | None = None,
        blocked: dict[str, set[str]] | None = None,
    ):
        if cascade is Cascade.FULL and llm is None:
            raise ValueError("Cascade.FULL needs an llm for the judge step")

        # Sorted by id so row order is stable and matches the embedding cache.
        self._entities = sorted(entities, key=lambda entity: entity.id)
        self._by_id = {entity.id: entity for entity in self._entities}
        self._embedder = embedder
        self._llm = llm
        self._cache_path = cache_path
        self._person_context = context  # hints from context.md for the judge prompts
        self._fuzzy_min = settings.get("resolve.fuzzy_min")
        self._embed_top_k = settings.get("resolve.embed_top_k")
        self._review_below = settings.get("review.confidence_threshold")
        self._matrix: np.ndarray | None = None

        # Human name rules (see name_rules), keyed by normalised text: an override is answered before the cascade;
        # a blocked id is never returned for that text.
        self._overrides = {_normalize(text): entity_id for text, entity_id in (overrides or {}).items()}
        self._blocked = {_normalize(text): ids for text, ids in (blocked or {}).items()}
        self.seen: set[str] = set()  # normalised texts asked about, for mark_rules_used

        # Same mention, same answer: one cascade per distinct (text, type, k); judge verdicts persist beside the
        # embedding cache so rebuilds make no repeat LLM calls (e.g. one tag seen in 900 documents).
        self._memo: dict[tuple, ResolveResult] = {}
        self._verdicts: dict[str, dict] | None = None
        self._verdict_path = cache_path.with_name(_VERDICT_FILE) if cache_path else None
        self._full = cascade is Cascade.FULL
        self._batch_size = settings.get("resolve.judge_batch")
        self._workers = settings.get("resolve.judge_workers")
        self._queries: dict[str, np.ndarray] = {}
        self._context: dict[str, str] = {}  # where a name was seen, shown to the batch judge
        self.judge_failures: list[str] = []  # names whose batch reply stayed unusable; listed in REVIEW.md

        self._steps = [self._exact, self._alias]
        if cascade is Cascade.FULL:
            self._steps += [self._fuzzy, self._embedding]

        # Lookup tables, e.g. "Widget 0" -> "widget 0" -> [Widget Zero].
        self._by_name: dict[str, list[Entity]] = {}
        self._by_alias: dict[str, list[Entity]] = {}
        self._fuzzy_names: list[tuple[str, Entity]] = []
        for entity in self._entities:
            self._by_name.setdefault(entity.canonical_name, []).append(entity)
            for name in [entity.canonical_name, *entity.aliases]:
                key = _normalize(name)
                self._by_alias.setdefault(key, []).append(entity)
                self._fuzzy_names.append((key, entity))

    def resolve(self, text: str, type: str | None = None, k: int = _DEFAULT_K) -> ResolveResult:
        self.seen.add(_normalize(text))
        key = (text, type, k)
        if key not in self._memo:
            self._memo[key] = self._cascade(text, type, k)
        return self._memo[key]

    def prefetch(self, requests: list[tuple[str, str | None]], context: dict[str, str] | None = None) -> None:
        """Resolve many (text, type) pairs up front: one embedding call, judge calls batched and in parallel.

        `context` maps a name to where it was seen (e.g. its sentence), so the judge can tell "widget" in a
        sentence about one product from another. Later resolve() calls answer from the cache.
        """
        self._context.update(context or {})
        local_steps = [self._exact, self._alias] + ([self._fuzzy] if self._full else [])
        pending = [
            (text, type) for text, type in dict.fromkeys(requests)
            if (text, type, _DEFAULT_K) not in self._memo and self._overridden(text) is None
            and not any(self._allowed(text, step(text, type)) for step in local_steps)
        ]
        if not self._full or not pending:
            return

        # One embedding call for every distinct text still unresolved.
        texts = list(dict.fromkeys(text for text, _ in pending))
        self._queries.update(zip(texts, self._embedder.embed_queries(texts)))

        if self._verdicts is None:
            self._verdicts = self._load_verdicts()
        items = []
        for text, type in pending:
            candidates = self._shortlist(text, type)
            key = _verdict_key(text, candidates)
            if candidates and key not in self._verdicts and key not in {item[0] for item in items}:
                items.append((key, text, candidates))

        batches = [items[start:start + self._batch_size] for start in range(0, len(items), self._batch_size)]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            replies = list(pool.map(self._judge_batch, batches))

        # Store and log in this thread; a failed batch is not logged, so a rebuild asks again.
        for batch, reply in zip(batches, replies):
            for index, (key, text, _) in enumerate(batch, start=1):
                if reply is None:
                    self._verdicts[key] = None
                    self.judge_failures.append(text)
                    continue
                self._remember(key, reply.get(f"N{index}"))

        for text, type in pending:
            self.resolve(text, type)

    def mark_rules_used(self, decisions: Decisions) -> None:
        """Mark the name_match decisions whose text was asked about (and whose forced entity exists) as used."""
        for decision in decisions.of(Kind.NAME_MATCH):
            text = _normalize(decision.target["text"])
            forced = decision.fix.get("entity_id")
            if text in self.seen and (forced is None or forced in self._by_id):
                decisions.mark_used(decision)

    def _overridden(self, text: str) -> Match | None:
        entity = self._by_id.get(self._overrides.get(_normalize(text)))
        if entity is None:
            return None
        return Match(entity.id, entity.canonical_name, entity.type, _EXACT_SCORE, Method.HUMAN, needs_review=False)

    def _allowed(self, text: str, matches: list[Match]) -> list[Match]:
        blocked = self._blocked.get(_normalize(text), set())
        return [match for match in matches if match.id not in blocked]

    def _cascade(self, text: str, type: str | None, k: int) -> ResolveResult:
        forced = self._overridden(text)
        if forced:
            return ResolveResult(Status.MATCHED, [forced], [])

        for step in self._steps:
            matches = self._allowed(text, step(text, type))
            if not matches:
                continue
            status = Status.MATCHED if len(matches) == 1 else Status.AMBIGUOUS
            return ResolveResult(status, matches[:k], [])

        return ResolveResult(Status.NO_MATCH, [], self._suggest(text, type, k))

    def _exact(self, text: str, type: str | None) -> list[Match]:
        found = _of_type(self._by_name.get(text, []), type)
        return [self._match(entity, _EXACT_SCORE, Method.EXACT) for entity in found]

    def _alias(self, text: str, type: str | None) -> list[Match]:
        found = _of_type(self._by_alias.get(_normalize(text), []), type)
        return [self._match(entity, _EXACT_SCORE, Method.ALIAS) for entity in found]

    def _fuzzy(self, text: str, type: str | None) -> list[Match]:
        ranked = self._fuzzy_ranked(text, type)
        if not ranked or ranked[0][0] < self._fuzzy_min:
            return []

        # Entities tying at the best score are all returned (ambiguous).
        best = ranked[0][0]
        tied = [(score, entity) for score, entity in ranked if score == best]
        return [self._match(entity, score / _PERCENT, Method.FUZZY) for score, entity in tied]

    def _embedding(self, text: str, type: str | None) -> list[Match]:
        # Nearest entities to the mention; the judge picks one or none.
        candidates = self._shortlist(text, type)
        if not candidates:
            return []

        verdict = self._judge(text, candidates)
        if verdict is None:
            return []
        entity, confidence = verdict
        return [self._match(entity, confidence, Method.EMBEDDING)]

    def _shortlist(self, text: str, type: str | None) -> list[Entity]:
        """Top-k catalog entities of `type` nearest to `text` by embedding (query vector cached by prefetch)."""
        blocked = self._blocked.get(_normalize(text), set())
        rows = [
            row for row, entity in enumerate(self._entities) if type in (None, entity.type) and entity.id not in blocked
        ]
        if not rows:
            return []
        if text not in self._queries:
            self._queries[text] = self._embedder.embed_queries([text])[0]
        nearest = cosine_top_k(self._queries[text], self._entity_matrix()[rows], self._embed_top_k)
        return [self._entities[rows[index]] for index, _ in nearest]

    def _judge_batch(self, batch: list[tuple[str, str, list[Entity]]]) -> dict | None:
        """One prompt for up to `judge_batch` mentions, each with its own candidates; None when the reply stays unusable."""
        blocks = []
        for index, (_, text, candidates) in enumerate(batch, start=1):
            lines = [f"  - {e.id}: {e.canonical_name} ({e.type}), aliases: {', '.join(e.aliases) or 'none'}" for e in candidates]
            seen = f"\n  Seen in: {self._context[text]}" if text in self._context else ""
            blocks.append("\n".join([f"[N{index}] Mention: {text}{seen}", *lines]))
        prompt = "\n".join([
            f"[task: {_TASK_JUDGE_BATCH}]",
            "For each mention, pick the candidate it refers to from ITS OWN list, or null if none is clearly the same thing.",
            "Use the 'Seen in' text to tell apart candidates with similar names.",
            _CONTEXT_BLOCK.format(context=self._person_context),
            *blocks,
            'Reply with JSON only: {"N1": {"id": "<candidate id or null>", "confidence": <0 to 1>}, "N2": ...}',
        ])
        for _ in range(_BATCH_ATTEMPTS):
            try:
                data = extract_json(self._llm.complete(prompt).text)
            except LLMError:
                continue
            if isinstance(data, dict):
                return data
        return None

    def _remember(self, key: str, reply) -> None:
        """Keep one judge reply in memory and in the verdict log."""
        self._verdicts[key] = reply
        if self._verdict_path is None:
            return

        # The catalog hash is recorded for the record only; it does not gate reuse.
        line = json.dumps({"catalog": self._catalog_hash(), "key": key, "reply": reply}) + "\n"
        # A line cut short by Ctrl+C has no newline: start a fresh line so this one stays readable.
        cut_short = self._verdict_path.exists() and not self._verdict_path.read_bytes().endswith(b"\n")
        with self._verdict_path.open("a") as log:
            log.write(("\n" if cut_short else "") + line)

    def _judge(self, text: str, candidates: list[Entity]) -> tuple[Entity, float] | None:
        lines = [f"- {e.id}: {e.canonical_name} ({e.type}), aliases: {', '.join(e.aliases) or 'none'}" for e in candidates]
        prompt = "\n".join(
            [
                f"[task: {_TASK_JUDGE}]",
                "Pick the candidate that the mention refers to, or none if no candidate is clearly the same thing.",
                _CONTEXT_BLOCK.format(context=self._person_context),
                f"Mention: {text}",
                "Candidates:",
                *lines,
                'Reply with JSON only: {"id": "<candidate id or null>", "confidence": <0 to 1>}',
            ]
        )

        data = self._verdict(text, candidates, prompt)
        if not isinstance(data, dict):
            return None

        # An id outside the candidates, or no usable confidence, counts as none.
        by_id = {entity.id: entity for entity in candidates}
        entity = by_id.get(data.get("id")) if isinstance(data.get("id"), str) else None
        confidence = data.get("confidence")
        if entity is None or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        return entity, min(max(float(confidence), 0.0), 1.0)

    def _verdict(self, text: str, candidates: list[Entity], prompt: str):
        """The judge's parsed reply, from the on-disk verdict log when this mention + shortlist was judged before."""
        if self._verdicts is None:
            self._verdicts = self._load_verdicts()

        key = _verdict_key(text, candidates)
        if key in self._verdicts:
            return self._verdicts[key]

        data = extract_json(self._llm.complete(prompt).text)
        self._remember(key, data)
        return data

    def _load_verdicts(self) -> dict[str, dict]:
        """Verdicts from the log, whatever the catalog then; a line that is not valid JSON is skipped."""
        if self._verdict_path is None or not self._verdict_path.exists():
            return {}

        verdicts = {}
        for line in self._verdict_path.read_text(errors="replace").splitlines():
            row = _parse_verdict_row(line)
            if row:
                verdicts[row["key"]] = row["reply"]
        return verdicts

    def _fuzzy_ranked(self, text: str, type: str | None) -> list[tuple[float, Entity]]:
        """(token_sort_ratio, entity), best name per entity, best first; e.g. (95.65, Widget Zero)."""
        key = _normalize(text)
        best: dict[str, float] = {}
        for name, entity in self._fuzzy_names:
            if type not in (None, entity.type):
                continue
            best[entity.id] = max(best.get(entity.id, 0.0), fuzz.token_sort_ratio(key, name))

        by_id = {entity.id: entity for entity in self._entities}
        ranked = [(score, by_id[id_]) for id_, score in best.items() if score > 0]
        return sorted(ranked, key=lambda pair: -pair[0])

    def _suggest(self, text: str, type: str | None, k: int) -> list[str]:
        return [entity.canonical_name for _, entity in self._fuzzy_ranked(text, type)[:k]]

    def _match(self, entity: Entity, score: float, method: Method) -> Match:
        needs_review = entity.needs_review or score < self._review_below
        return Match(entity.id, entity.canonical_name, entity.type, score, method, needs_review)

    def _entity_matrix(self) -> np.ndarray:
        """One embedding per entity (name + aliases), cached on disk keyed by the catalog hash."""
        if self._matrix is not None:
            return self._matrix

        if self._cache_path is None:
            self._matrix = self._embed_entities()
            return self._matrix

        catalog_hash = self._catalog_hash()
        sidecar = self._cache_path.with_name(self._cache_path.name + ".sha256")
        if self._cache_path.exists() and sidecar.exists() and sidecar.read_text() == catalog_hash:
            self._matrix = _read_matrix(self._cache_path)
        if self._matrix is not None:
            return self._matrix

        self._matrix = self._embed_entities()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("wb") as handle:
            np.save(handle, self._matrix)
        sidecar.write_text(catalog_hash)
        return self._matrix

    def _embed_entities(self) -> np.ndarray:
        texts = [", ".join([entity.canonical_name, *entity.aliases]) for entity in self._entities]
        return self._embedder.embed_passages(texts)

    def _catalog_hash(self) -> str:
        rows = [[e.id, e.canonical_name, sorted(e.aliases)] for e in self._entities]
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def name_rules(decisions: Decisions) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(overrides, blocked) from the `name_match` decisions: {text: entity id to always use}, {text: ids never used}."""
    overrides: dict[str, str] = {}
    blocked: dict[str, set[str]] = {}
    for decision in decisions.of(Kind.NAME_MATCH):
        text = decision.target["text"]
        if decision.fix.get("entity_id"):
            overrides[text] = decision.fix["entity_id"]
        if decision.fix.get("not_entity_id"):
            blocked.setdefault(text, set()).add(decision.fix["not_entity_id"])
    return overrides, blocked


def _read_matrix(path: Path) -> np.ndarray | None:
    """The saved embeddings, or None when the file is unreadable (e.g. cut short by Ctrl+C): a miss."""
    try:
        return np.load(path)
    except (OSError, ValueError, EOFError):
        return None


def _parse_verdict_row(line: str) -> dict | None:
    """The {key, reply, ...} row of one log line; None for a blank, cut-short or malformed line."""
    try:
        row = json.loads(line)
    except ValueError:
        return None
    if not isinstance(row, dict) or "key" not in row or "reply" not in row:
        return None
    return row


def _verdict_key(text: str, candidates: list[Entity]) -> str:
    return text + "|" + ",".join(entity.id for entity in candidates)


def _normalize(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed: " SPR_Zero. " -> "spr zero"."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _of_type(entities: list[Entity], type: str | None) -> list[Entity]:
    return [entity for entity in entities if type in (None, entity.type)]
