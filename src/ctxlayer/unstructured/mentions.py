"""Mentions by catalog-name matching: which known entities each document names, without an LLM."""
import re
import unicodedata

from ctxlayer.common.catalog import Entity
from ctxlayer.unstructured.documents import Document

_MIN_NAME_CHARS = 3
_SCORE = 1.0
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Generic English words that are also short entity names somewhere ("May", "Will"); never mentions on their own.
_FUNCTION_WORDS = frozenset({
    "the", "and", "for", "but", "nor", "yet", "may", "will", "can", "could", "would", "should", "shall",
    "must", "might", "new", "all", "any", "one", "two", "per", "via", "from", "with", "into", "over",
    "under", "out", "off", "more", "most", "less", "than", "this", "that", "these", "those", "also",
})


def find_mentions(docs: list[Document], entities: list[Entity]) -> dict[str, list[dict]]:
    """{doc_id: [{name, entity_id, score}]} for catalog names / aliases found as whole words in the text.

    Skipped: names under 3 characters, names without a letter, generic function words ("may"), names shared
    by two entities (ambiguous) and entities flagged for review. E.g. "strike in Region One" -> [{"name": "region one", "entity_id": "location:region_one"}].
    """
    owners = _name_owners(entities)
    if not owners:
        return {doc.doc_id: [] for doc in docs}

    # One alternation, longest names first, padded with spaces so matches are whole words.
    names = sorted(owners, key=len, reverse=True)
    pattern = re.compile(" (" + "|".join(re.escape(name) for name in names) + ") ")

    found = {}
    for doc in docs:
        text = " " + _normalize(f"{doc.title} {doc.summary} {doc.body}") + " "
        # One entry per entity, whichever of its names matched first alphabetically.
        by_entity = {}
        for name in sorted(_matches(text, pattern)):
            by_entity.setdefault(owners[name], name)
        found[doc.doc_id] = [
            {"name": name, "entity_id": entity_id, "score": _SCORE} for entity_id, name in sorted(by_entity.items())
        ]
    return found


def _name_owners(entities: list[Entity]) -> dict[str, str]:
    """Normalized name -> the single trusted entity it belongs to."""
    owners: dict[str, set[str]] = {}
    for entity in entities:
        if entity.needs_review:
            continue
        for name in {_normalize(text) for text in [entity.canonical_name, *entity.aliases]}:
            if len(name) >= _MIN_NAME_CHARS and any(char.isalpha() for char in name) and name not in _FUNCTION_WORDS:
                owners.setdefault(name, set()).add(entity.id)
    return {name: ids.pop() for name, ids in owners.items() if len(ids) == 1}


def _matches(text: str, pattern: re.Pattern) -> set[str]:
    """Every match, including neighbours that share one space ("region one region two" -> both)."""
    hits, position = set(), 0
    while True:
        match = pattern.search(text, position)
        if not match:
            return hits
        hits.add(match.group(1))
        position = match.end() - 1


def _normalize(text: str) -> str:
    """Accents stripped, lowercase, punctuation to spaces: "São Vale's" -> "sao vale s"."""
    plain = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return _NON_ALNUM.sub(" ", plain.lower()).strip()
