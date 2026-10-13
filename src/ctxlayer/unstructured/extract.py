"""Event-type induction and per-document LLM extraction over numbered sentences, with a content-hash cache."""
import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ctxlayer.common.files import write_json
from ctxlayer.common.llm import DEFAULT_MODEL_NAME, LLM, LLMError, extract_json
from ctxlayer.common.runlog import progress
from ctxlayer.unstructured.documents import Document

_PROMPT_VERSION = "extract-v4"
_STATUSES = ("reported", "announced", "forecast")  # happened / declared for later / expected or speculated
_REPORTED = "reported"
_DATE_STATED = "stated"            # the document gives the event's own date
_DATE_PUBLICATION = "publication"  # no event date given: the document's date stands in
MIN_AFFECT_CONFIDENCE = 0.6  # 0.9 = stated outright, 0.6 = implied; anything lower is not an effect
_MAX_EVENTS = 6
_SUMMARY_WORDS = 20
# Sentence end: . ! or ? followed by space and a capital, digit or opening quote/bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_SAMPLE_SEED = 0
_ATTEMPTS = 2  # first call + one retry with the validation errors appended
_UNKNOWN = "unknown"
_OTHER = "other"
_OTHER_TYPE = {"name": _OTHER, "description": "Any event that fits none of the other types."}

_EVENT_TYPES_PROMPT = """[task: event_types]
Below are summaries of documents from one collection, and the entity types of its database.
Propose 8 to 12 event types that cover the kinds of real-world events these documents report.
Types must be generic to the collection, not tied to one document.

Entity types: {entity_types}

Additional collection context:
{context}

Document summaries:
{summaries}

Reply with JSON only: {{"event_types": [{{"name": "snake_case_name", "description": "one sentence"}}]}}"""

_EXTRACT_PROMPT = """[task: extract]
Extract the events this document reports. The text is split into numbered sentences [S1], [S2], ...
Rules:
- Report only what a sentence states. If the document reports no real event (opinion, speculation, background),
  reply {{"events":[]}}. An empty answer is correct when nothing happened.
- Refer to evidence by sentence numbers only ("s": [3]); never copy text.
- At most {max_events} events, most important first. "summary" at most {summary_words} words.
- Event "type" must be one of the event types listed; use "other" if none fits.
- "affects": only entities whose change a cited sentence states; leave the list empty when no change is stated.
  "aspect" is what changes (e.g. "price", "output", "demand", "cost").
  "direction" is "up" or "down" only when that sentence says so; "unknown" when the effect is implied or hedged.
  "confidence": 0.9 when stated outright, 0.6 when implied; do not list weaker effects.
- "status": "reported" (it happened), "announced" (declared for later, e.g. a planned increase),
  "forecast" (expected or speculated).
- "date": the date the event itself happened or takes effect, as ISO (YYYY-MM-DD), only if the text gives it;
  otherwise "". Never copy the document date for an older or later event.
- Reply with one line of compact JSON, no code fence, no indentation.

Examples (a made-up domain; TYPE stands for one of the listed event types):
Example: "[S1] Storms closed the Port of Avel for a week. [S2] Gizmo prices rose 8% as cargo stalled."
-> {{"events":[{{"type":"TYPE","date":"","status":"reported","summary":"Storms close the Port of Avel","locations":["Port of Avel"],"organizations":[],"affects":[{{"entity":"Gizmo","aspect":"price","direction":"up","confidence":0.9,"s":[2]}}],"s":[1]}}]}}
Example: "[S1] Analysts say widget demand could shift if interest rates change."
-> {{"events":[]}}
Example: "[S1] A new plant opened in Brava. [S2] It will make gadgets."
-> {{"events":[{{"type":"TYPE","date":"","status":"reported","summary":"Gadget plant opens in Brava","locations":["Brava"],"organizations":[],"affects":[],"s":[1]}}]}}
Example: "[S1] Dry weather may tighten sprocket supply later this year."
-> {{"events":[{{"type":"TYPE","date":"","status":"forecast","summary":"Dry weather may tighten sprocket supply","locations":[],"organizations":[],"affects":[{{"entity":"Sprocket","aspect":"supply","direction":"unknown","confidence":0.6,"s":[1]}}],"s":[1]}}]}}

Event types:
{event_types}

Entity types: {entity_types}

Additional collection context:
{context}

Reply format:
{{"events":[{{"type":"","date":"","status":"reported|announced|forecast","summary":"","locations":[""],"organizations":[""],"affects":[{{"entity":"","aspect":"","direction":"up|down|unknown","confidence":0.0,"s":[1]}}],"s":[1]}}]}}

Document (date {date}):
{sentences}"""

_RETRY_NOTE = "\n\nYour previous reply was invalid:\n{errors}\nReply again with corrected JSON only."


class _EventTypeReply(BaseModel):
    event_types: list[dict[str, str]]


class _Affect(BaseModel):
    entity: str
    aspect: str = ""
    direction: Literal["up", "down", "unknown"] = _UNKNOWN
    confidence: float = Field(0.0, ge=0, le=1)
    quote: str = ""


class _ExtractedEvent(BaseModel):
    type: str
    date: str | None = None
    date_source: str = _DATE_STATED
    status: str = _REPORTED
    summary: str
    locations: list[str] = []
    organizations: list[str] = []
    affects: list[_Affect] = []
    evidence: str
    needs_review: bool = False


class Extraction(BaseModel):
    events: list[_ExtractedEvent] = []


def _loose_ids(value) -> list:
    """Sentence ids as the model may write them: 2, "2", "S2", "[S2]", ["S2", 3] -> [2] / [2, 3]."""
    items = value if isinstance(value, list) else [value]
    ids = []
    for item in items:
        found = None if isinstance(item, bool) else re.search(r"-?\d+", str(item))
        if found:
            ids.append(int(found.group()))  # -1 stays -1 and is rejected later as out of range
    return ids


class _AffectReply(BaseModel):
    entity: str
    aspect: str = ""
    direction: Literal["up", "down", "unknown"] = _UNKNOWN
    confidence: float = Field(0.0, ge=0, le=1)
    s: list[int] = []

    _ids = field_validator("s", mode="before")(lambda value: _loose_ids(value))


class _EventReply(BaseModel):
    type: str
    date: str | None = None
    status: str = _REPORTED
    summary: str
    locations: list[str] = []
    organizations: list[str] = []
    affects: list[_AffectReply] = []
    s: list[int] = []

    _ids = field_validator("s", mode="before")(lambda value: _loose_ids(value))


class _Reply(BaseModel):
    events: list[_EventReply] = []


def induce_event_types(docs: list[Document], llm: LLM, ontology: dict, sample: int, context: str = "") -> list[dict]:
    """[{name, description}] from a seeded sample of summaries, always ending with "other"."""
    # Seeded so the same documents give the same prompt on every build.
    summaries = [doc.summary for doc in docs if doc.summary]
    chosen = random.Random(_SAMPLE_SEED).sample(summaries, min(sample, len(summaries)))

    prompt = _EVENT_TYPES_PROMPT.format(
        entity_types=", ".join(_entity_types(ontology)),
        context=context,
        summaries="\n".join(f"- {text}" for text in chosen),
    )
    reply = None
    for _ in range(_ATTEMPTS):
        try:
            reply = _EventTypeReply.model_validate(extract_json(llm.complete(prompt).text))
            break
        except (LLMError, ValueError):
            continue
    items = reply.event_types if reply else []

    types = [{"name": item["name"], "description": item.get("description", "")} for item in items if item.get("name")]
    if _OTHER not in {item["name"] for item in types}:
        types.append(dict(_OTHER_TYPE))
    return types


def extract_document(doc: Document, llm: LLM, event_types: list[dict], entity_types: list[str]) -> Extraction | None:
    """Validated extraction, or None when the reply stays invalid after one retry."""
    extraction, _ = _extract_or_error(doc, llm, event_types, entity_types)
    return extraction


def extract_all(
    docs: list[Document], llm: LLM, ontology: dict, cache_dir: Path, workers: int, context: str = "",
    stats: Counter | None = None,
) -> tuple[dict[str, Extraction], list[dict]]:
    """(extractions by doc_id, [{doc_id, error}]); cached documents make no LLM call.

    `stats`, when given, gets `hits` (documents answered from the cache) and `misses` added.
    """
    event_types = ontology["event_types"]
    entity_types = _entity_types(ontology)
    model = getattr(llm, "model_name", DEFAULT_MODEL_NAME)

    def run(doc: Document) -> tuple[Extraction | None, str, bool]:
        prompt = _prompt(doc, event_types, entity_types, context)
        cache_path = cache_dir / f"{_cache_key(prompt, model)}.json"
        cached = _read_cached(cache_path)
        if cached:
            return cached, "", True

        # Entries made before the model joined the key are still good: use one and copy it to the new key.
        cached = _read_cached(cache_dir / f"{_cache_key(prompt)}.json")
        if cached:
            write_json(cache_path, cached.model_dump())
            return cached, "", True

        extraction, error = _extract_or_error(doc, llm, event_types, entity_types, context)
        if extraction is not None:
            write_json(cache_path, extraction.model_dump())
        return extraction, error, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = []
        for completed, outcome in enumerate(pool.map(run, docs), start=1):
            outcomes.append(outcome)
            progress("build_unstructured.extract", completed, len(docs))

    if stats is not None:
        hits = sum(cached for _, _, cached in outcomes)
        stats["hits"] += hits
        stats["misses"] += len(outcomes) - hits

    results, failures = {}, []
    for doc, (extraction, error, _) in zip(docs, outcomes):
        if extraction is None:
            failures.append({"doc_id": doc.doc_id, "error": error})
            continue
        results[doc.doc_id] = extraction
    return results, failures


def _entity_types(ontology: dict) -> list[str]:
    return [item["name"] for item in ontology["types"]]


def _read_cached(path: Path) -> Extraction | None:
    """The saved extraction, or None when missing or unreadable (e.g. cut short by Ctrl+C): a miss."""
    try:
        return Extraction.model_validate(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def _cache_key(prompt: str, model: str = "") -> str:
    """Hash of the full prompt: new event types, entity types, document text or model mean a new extraction.

    An empty `model` gives the key of caches written before the model joined the key.
    """
    return hashlib.sha256((_PROMPT_VERSION + model + prompt).encode()).hexdigest()


def _prompt(doc: Document, event_types: list[dict], entity_types: list[str], context: str = "") -> str:
    numbered = "\n".join(f"[S{index}] {sentence}" for index, sentence in enumerate(_sentences(doc), start=1))
    return _EXTRACT_PROMPT.format(
        event_types="\n".join(f"- {item['name']}: {item['description']}" for item in event_types),
        entity_types=", ".join(entity_types), context=context, date=doc.date, sentences=numbered,
        max_events=_MAX_EVENTS, summary_words=_SUMMARY_WORDS,
    )


def _sentences(doc: Document) -> list[str]:
    """Title, summary and body split into sentences, e.g. "Drought hits. Mills close." -> ["Drought hits.", "Mills close."]."""
    sentences = []
    for part in (doc.title, doc.summary, doc.body):
        for paragraph in part.splitlines():
            sentences += [" ".join(item.split()) for item in _SENTENCE_END.split(paragraph) if item.strip()]
    return sentences


def _iso_day(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None



def _extract_or_error(
    doc: Document, llm: LLM, event_types: list[dict], entity_types: list[str], context: str = "",
) -> tuple[Extraction | None, str]:
    prompt = _prompt(doc, event_types, entity_types, context)
    known_types = {item["name"] for item in event_types}

    # A failed attempt feeds its error back to the model once.
    error = ""
    for _ in range(_ATTEMPTS):
        attempt = prompt + (_RETRY_NOTE.format(errors=error) if error else "")
        try:
            reply = llm.complete(attempt)
            parsed = extract_json(reply.text)
            if parsed is None:
                raise ValueError("reply contains no JSON")
            reply = _Reply.model_validate(parsed)
            if reply.events and not any(event.s for event in reply.events):
                # The model quoted text instead of citing numbers: retry rather than drop every event silently.
                raise ValueError('events must cite sentence numbers in "s", e.g. "s": [3]')
            return _clean(reply, doc, known_types), ""
        except (LLMError, ValueError) as exc:
            error = str(exc)
    return None, error


def _clean(reply: _Reply, doc: Document, known_types: set[str]) -> Extraction:
    """Sentence ids -> exact text; events without a valid id are dropped; up/down kept only with a valid id."""
    sentences = _sentences(doc)

    def text_of(ids: list[int]) -> str:
        # Ids are 1-based; out-of-range ids are ignored, e.g. [3, 99] with 10 sentences -> sentence 3 only.
        return " ".join(sentences[i - 1] for i in dict.fromkeys(ids) if 1 <= i <= len(sentences))

    events = []
    for item in reply.events[:_MAX_EVENTS]:
        evidence = text_of(item.s)
        if not evidence:
            continue

        event = _ExtractedEvent(
            type=item.type, date=item.date, status=item.status,
            summary=" ".join(item.summary.split()[:_SUMMARY_WORDS]),
            locations=item.locations, organizations=item.organizations, evidence=evidence,
            # Weaker than "implied" is not an effect: drop it rather than store a guess.
            affects=[
                _affect(affect, text_of(affect.s)) for affect in item.affects
                if affect.confidence >= MIN_AFFECT_CONFIDENCE
            ],
        )
        if event.type not in known_types:
            event.type = _OTHER
            event.needs_review = True

        # Unknown status: assume it happened, and ask a human.
        if event.status not in _STATUSES:
            event.status = _REPORTED
            event.needs_review = True

        # The model's date must be an ISO day ("2025-05-05", time part dropped). None given: the document date
        # stands in and date_source says so; an unreadable one is also flagged.
        day = _iso_day(event.date)
        if event.date and day is None:
            event.needs_review = True
        event.date_source = _DATE_STATED if day else _DATE_PUBLICATION
        event.date = day or doc.date
        events.append(event)

    return Extraction(events=events)


def _affect(affect: _AffectReply, quote: str) -> _Affect:
    direction = affect.direction if quote else _UNKNOWN
    return _Affect(
        entity=affect.entity, aspect=affect.aspect, direction=direction, confidence=affect.confidence, quote=quote,
    )
