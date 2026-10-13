"""Catalog-name mentions: whole words, ambiguous and short names skipped, flagged entities skipped."""
from ctxlayer.common.catalog import Entity
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.mentions import find_mentions


def _doc(text: str) -> Document:
    return Document("D1", "2025-01-01", "", "", text)


def _ids(text: str, entities: list[Entity]) -> list[str]:
    return [item["entity_id"] for item in find_mentions([_doc(text)], entities)["D1"]]


def test_names_and_aliases_match_as_whole_words_any_case():
    entities = [
        Entity("place:north_vale", "Location", "North Vale", aliases=["NV Region"]),
        Entity("item:gadget", "Item", "Gadget"),
    ]

    assert _ids("Strike in north-vale hits GADGET output; nv region quiet.", entities) == ["item:gadget", "place:north_vale"]
    assert _ids("Gadgets are fine.", entities) == []  # no partial-word match


def test_ambiguous_short_and_flagged_names_are_skipped():
    entities = [
        Entity("item:one", "Item", "Widget", aliases=["WX"]),       # "wx" is too short
        Entity("item:two", "Item", "Gizmo", aliases=["Widget"]),    # "widget" now shared -> ambiguous
        Entity("item:three", "Item", "Doohickey", needs_review=True),
    ]

    assert _ids("Widget, WX and Doohickey rose; Gizmo fell.", entities) == ["item:two"]

