"""Linking bridge: candidate drivers of each move from events and tagged documents, and the explain_move tool."""
import math
from datetime import date

from ctxlayer.common.catalog import Entity
from ctxlayer.common.config import Settings
from ctxlayer.linking._series import raw_ids
from ctxlayer.structured.compiler import MetricRequest, run_metric
from ctxlayer.structured.metrics import Metric
from ctxlayer.unstructured.graph_store import GraphStore, ReviewFilter
from ctxlayer.unstructured._names import DOC_PREFIX

_AFFECTS = "AFFECTS"
_DOC_RELATIONS = ("TAGGED_WITH", "MENTIONS")
_HAS_PARENT = "HAS_PARENT"
_PART_OF = "PART_OF"
_DOCUMENT = "Document"
_MATCH, _UNKNOWN, _OPPOSITE = "match", "unknown", "opposite"
_NO_DRIVERS = "No candidate drivers found for this period: no event or tagged document in the preceding days matches."
_NO_MOVES = "No moves in this period, so there is nothing to explain."


def candidate_drivers(
    moves: list[dict], store: GraphStore, events: list[dict], catalog: list[Entity], settings: Settings,
    window: tuple[str, str],
) -> list[dict]:
    """Top-N candidate drivers per move inside the document `window`.

    Each move carries `catalog_id` (its entity in the catalog). Candidates are events dated 0..lookback_days before
    the move that AFFECT the entity or an ancestor, or whose documents are tagged with / mention one of them,
    in an overlapping location. score = confidence x direction weight x exp(-lag / decay) x relation weight.
    Ranking features only: a high score is never evidence of causality.
    """
    config = {**settings.get("linking"), "min_confidence": settings.get("review.confidence_threshold")}
    by_id = {entity.id: entity for entity in catalog}

    # Flagged events never drive; each event's places (with PART_OF ancestors) are computed once, not per move.
    trusted = [event for event in events if not event.get("needs_review")]
    event_places = {event["event_id"]: _closure_of(store, event["location_ids"], _PART_OF) for event in trusted}

    rows = []
    for move in moves:
        entity = by_id.get(move.get("catalog_id"))
        if entity is None or not window[0] <= move["date"] <= window[1]:
            continue

        targets = _closure(store, entity.id, _HAS_PARENT)
        places = entity.attributes.get("locations", [])
        move_places = (set(places), _closure_of(store, places, _PART_OF))
        scored = []
        for event in trusted:
            row = _score_event(move, event, targets, move_places, event_places[event["event_id"]], store, config)
            if row:
                scored.append(row)
        scored.sort(key=lambda row: (-row["score"], row["event_id"]))
        rows += scored[: config["top_n"]]
    return rows


def explain_move(
    con_sem, semantic: dict, store: GraphStore, links: list[dict], moves: list[dict],
    entity_id: str, start: str | None = None, end: str | None = None, limit: int | None = None,
) -> dict:
    """{drivers, note?, coverage, moves, series} for a catalog entity: candidate drivers first, then the evidence.

    Drivers appear first because long series get cut by the agent's transcript limit. Outside the document window drivers
    stay empty and `note` names the coverage; no match says so explicitly. `limit` caps the series rows.
    """
    nodes = store.subgraph(entity_id, depth=0)["nodes"]
    if not nodes:
        return {"error": {"type": "unknown_entity", "message": f"{entity_id!r} is not in the graph"}}

    ids = raw_ids(con_sem, semantic, nodes[0]["props"].get("sources", []))
    if not ids:
        return {"error": {"type": "no_data", "message": f"{entity_id!r} has no rows in the measures views"}}

    request = MetricRequest(metric=Metric.SERIES, entities=[], start=start, end=end)
    series = run_metric(con_sem, request, semantic, ids, limit)
    if "error" in series:
        return series

    first, last = series["definition"]["window"]
    id_set = {str(item) for item in ids}
    shown = [move for move in moves if str(move["entity_id"]) in id_set and first <= move["date"] <= last]
    coverage = _doc_window(store)
    drivers, note = [], None
    if coverage is None:
        note = "There are no documents, so no candidate drivers can be found."
    elif last < coverage[0] or first > coverage[1]:
        note = f"No documents cover {first} to {last}; documents cover {coverage[0]} to {coverage[1]}."
    elif not shown:
        note = _NO_MOVES
    else:
        drivers = _drivers(store, links, shown)
        note = None if drivers else _NO_DRIVERS

    result = {"drivers": drivers, **({"note": note} if note else {}), "coverage": coverage, "moves": shown}
    result["series"] = series
    citations = [{"kind": "doc", "doc_id": doc} for driver in drivers for doc in driver["docs"]]
    result["_citations"] = [{"kind": "sql", "sql": series["sql"]}, *citations]
    return result


# ------------------------------------------------------------------ scoring

def _score_event(
    move: dict, event: dict, targets: set[str], move_places: tuple[set, set], event_places: set[str],
    store: GraphStore, config: dict,
) -> dict | None:
    if not event.get("date"):
        return None
    lag = (date.fromisoformat(move["date"]) - date.fromisoformat(event["date"])).days
    if lag < 0 or lag > config["lookback_days"]:
        return None

    # Low-confidence affects are review items, not runtime evidence.
    target_affects = [affect for affect in event["affects"] if affect["entity_id"] in targets]
    candidates = [
        {**_candidate(affect["confidence"], affect["direction"], _AFFECTS, affect["doc_id"], affect["quote"], move, config),
         "aspect": affect.get("aspect", ""), "direction": affect["direction"]}
        for affect in target_affects if affect["confidence"] >= config["min_confidence"]
    ]
    # Documents only link an event the event itself does not say affects the entity, e.g. a "down" affect stays opposite.
    if not target_affects:
        candidates = [
            _candidate(event["confidence"], _UNKNOWN, relation, doc_id, _first_quote(event), move, config)
            for doc_id in event["doc_ids"] for relation in _tagged(store, doc_id, targets)
        ]
    if not candidates or not _overlaps(event["location_ids"], event_places, move_places):
        return None

    decay = math.exp(-lag / config["decay_days"])
    best = max(candidates, key=lambda item: item["weight"])
    return {
        "move_id": move["id"], "event_id": event["event_id"], "score": best["weight"] * decay, "lag_days": lag,
        "relation": best["relation"], "direction_match": best["direction_match"],
        "aspect": best.get("aspect", ""), "direction": best.get("direction", _UNKNOWN),
        "evidence_doc": best["doc_id"], "evidence_quote": best["quote"],
    }


def _candidate(
    confidence: float, direction: str, relation: str, doc_id: str, quote: str, move: dict, config: dict,
) -> dict:
    """One way an event can relate to a move; `weight` = confidence x direction weight x relation weight."""
    match = _direction_match(move["direction"], direction)
    weight = confidence * config["direction_weights"][match] * config["relation_weights"][relation]
    return {"weight": weight, "relation": relation, "direction_match": match, "doc_id": doc_id, "quote": quote}


def _direction_match(move_direction: str, affect_direction: str) -> str:
    """"match" when an affect goes the move's way, "opposite" when against it, else "unknown"."""
    if affect_direction == _UNKNOWN:
        return _UNKNOWN
    return _MATCH if affect_direction == move_direction else _OPPOSITE


# ---------------------------------------------------------------- the graph

def _closure(store: GraphStore, node_id: str, rel: str) -> set[str]:
    """`node_id` and everything reached by following `rel` edges outwards."""
    return _closure_of(store, [node_id], rel)


def _closure_of(store: GraphStore, node_ids: list[str], rel: str) -> set[str]:
    found = set(node_ids)
    pending = list(node_ids)
    while pending:
        current = pending.pop()
        for edge in store.neighbors(current, rel=rel, review=ReviewFilter.EXCLUDE):
            if edge["src"] != current or edge["dst"] in found:
                continue
            found.add(edge["dst"])
            pending.append(edge["dst"])
    return found


def _overlaps(event_ids: list[str], event_closure: set[str], move_places: tuple[set, set]) -> bool:
    """True when the event's places and the move's are the same or nested via PART_OF; no place on a side passes."""
    move_ids, move_closure = move_places
    if not event_ids or not move_ids:
        return True
    return bool(set(event_ids) & move_closure or move_ids & event_closure)


def _tagged(store: GraphStore, doc_id: str, targets: set[str]) -> set[str]:
    """Relations (TAGGED_WITH / MENTIONS) from a document to any of `targets`."""
    node_id = DOC_PREFIX + doc_id
    return {
        edge["rel"] for edge in store.neighbors(node_id, review=ReviewFilter.EXCLUDE)
        if edge["src"] == node_id and edge["rel"] in _DOC_RELATIONS and edge["dst"] in targets
    }


def _first_quote(event: dict) -> str:
    return event["quotes"][0] if event["quotes"] else ""


def _doc_window(store: GraphStore) -> list[str] | None:
    days = [node["props"]["date"] for node in store.nodes(_DOCUMENT)]
    return [min(days), max(days)] if days else None


def _drivers(store: GraphStore, links: list[dict], moves: list[dict]) -> list[dict]:
    drivers = []
    for move in sorted(moves, key=lambda item: item["date"]):
        own = sorted((link for link in links if link["move_id"] == move["id"]), key=lambda link: -link["score"])
        for link in own:
            # Links can outlive a rebuilt graph (build-unstructured alone): skip events that are gone.
            found = store.subgraph(link["event_id"], depth=0)["nodes"]
            if not found:
                continue
            event = found[0]
            drivers.append({
                "move_id": move["id"], "event": event["name"], "event_date": event["props"].get("date"),
                "event_status": event["props"].get("status", "reported"),
                "date_source": event["props"].get("date_source", "stated"),
                "score": link["score"], "lag_days": link["lag_days"], "relation": link["relation"],
                "direction_match": link["direction_match"], "aspect": link.get("aspect", ""),
                "direction": link.get("direction", _UNKNOWN), "docs": event["props"]["doc_ids"],
                "quote": link["evidence_quote"],
            })
    return drivers
