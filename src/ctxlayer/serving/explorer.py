"""Context explorer: one self-contained HTML page showing everything the build produced for a database.

Overview first (what was understood, headline counts), detail below (columns, rules, ontology, entities,
price moves with their candidate drivers, events, graph neighbours, review items). Everything is read
from the artifacts, so the page shows exactly what the agent's tools use. No network access: data is
embedded as JSON and rendered by a few lines of plain JavaScript.

Reviewable rows carry the `kind` and `target` of a human decision (spec section 8.2) and any decision
already recorded for them; the page lets a reviewer add more and export them as decisions.yaml.
"""
import json
from collections import defaultdict
from pathlib import Path

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.decisions import Decision, Decisions, Kind
from ctxlayer.common.files import read_jsonl, read_yaml
from ctxlayer.common.ontology import load_ontology
from ctxlayer.common.roles import ROLES

_OUT_FILE = "context_explorer.html"
_STATE_FILE = "build_state.json"
_DECISIONS_FILE = "decisions.yaml"
_MAX_MOVES_INSIDE = 300   # inside the news period, biggest change first
_MAX_MOVES_OUTSIDE = 100  # before or after it (no articles can support these)
_MAX_NEIGHBOURS = 60      # per entity in the graph explorer
_MAX_EVIDENCE = 3         # sentences shown per connection
_MAX_NAME_MATCHES = 300   # least sure first
_QUOTE_CHARS = 300
_ENTITY_TYPES_SKIPPED = {"Document", "Event", "Move", "Organization"}
_STALE = "stale"
_TAGGED_WITH = "TAGGED_WITH"
_MENTIONS = "MENTIONS"
_FROM_THE_DATA = {"HAS_PARENT", "AT_LOCATION", "PART_OF"}
# Connection kinds shown first: links backed by a sentence, then tag links, then links from the data.
_SENTENCE_TIER, _TAG_TIER, _DATA_TIER = 0, 1, 2
# Rule rows a reviewer can change: label shown, and the `field` of the rule target (spec 8.2).
_RULE_FIELDS = {
    "main measure": "primary_measure", "never aggregate across": "denominations",
    "scenario labels": "scenario.values", "time grains": "time.grains",
}


def write_explorer(settings: Settings, state: dict | None = None) -> Path:
    """Write artifacts/<database>/context_explorer.html and return its path.

    `state` is {"paused_after": phase or None, "phases": [{phase, state, reason}]} (the pipeline's phase
    status). Without it the page reads build_state.json for the pause and shows no stale list.
    """
    out_dir = artifacts_dir(settings)
    data = _collect(out_dir, settings, _page_state(out_dir, state))

    # "</" inside the JSON would end the <script> block early.
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    path = out_dir / _OUT_FILE
    path.write_text(_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------- collect

def _collect(out_dir: Path, settings: Settings, state: dict) -> dict:
    roles = _yaml(out_dir / "roles.yaml") or {"tables": {}}
    semantic = _yaml(out_dir / "semantic.yaml") or {"tables": {}}
    ontology = load_ontology(out_dir) if (out_dir / "ontology.yaml").is_file() else {}
    entities = _yaml(out_dir / "entities.yaml") or []
    text_entities = _yaml(out_dir / "graph/entities_text.yaml") or []
    manifest = _json(out_dir / "build_manifest.json") or {"stages": {}}
    nodes = _jsonl(out_dir / "graph/nodes.jsonl") + _jsonl(out_dir / "links/nodes.jsonl")
    edges = _jsonl(out_dir / "graph/edges.jsonl") + _jsonl(out_dir / "links/edges.jsonl")
    moves = _jsonl(out_dir / "structured/moves.jsonl")
    drivers = _jsonl(out_dir / "links/candidate_drivers.jsonl")
    review = (out_dir / "REVIEW.md").read_text() if (out_dir / "REVIEW.md").is_file() else ""
    decisions = Decisions.load(settings.root / settings.get("paths.review", "review") / _DECISIONS_FILE)

    by_id = {node["id"]: node for node in nodes}
    keys_of = {item["event_id"]: item.get("keys", []) for item in _jsonl(out_dir / "docs/events.jsonl")}
    all_entities = [*entities, *text_entities]
    events = [node for node in nodes if node["type"] == "Event"]
    documents = [node for node in nodes if node["type"] == "Document"]
    names = {item["id"]: item["canonical_name"] for item in all_entities}

    meta = _meta(settings, manifest, semantic)
    sorted_events = sorted(events, key=lambda n: n["props"].get("date") or "")
    return {
        "meta": meta,
        "counts": _counts(roles, all_entities, documents, events, moves, drivers, review, edges),
        "tables": _tables(roles, semantic, decisions),
        "rules": _rules(semantic, decisions),
        "ontology": {
            "types": [_type_row(item, decisions) for item in ontology.get("types", [])],
            "relations": ontology.get("relations", []),
            "event_types": [
                _reviewed(item, Kind.EVENT_TYPE, {"name": item["name"]}, decisions)
                for item in ontology.get("event_types", [])
            ],
        },
        "entities": [
            *(_reviewed(_entity(item), Kind.ENTITY, {"id": item["id"]}, decisions) for item in entities),
            *(_reviewed(_entity(item), Kind.NEW_ENTITY, {"id": item["id"]}, decisions) for item in text_entities),
        ],
        "name_matches": _name_matches(out_dir, names, decisions),
        "moves": _moves(moves, drivers, by_id, meta["document_window"], keys_of, decisions),
        "events": [_event(node, keys_of.get(node["id"], []), decisions) for node in sorted_events],
        "merged_events": _merged_events(sorted_events, keys_of, decisions),
        "neighbours": _neighbours(all_entities, edges, by_id),
        "review": review,
        "decisions": [_decision_dict(item) for kind in Kind for item in decisions.of(kind)],
        "state": state,
        "roles": ROLES,
    }


def _page_state(out_dir: Path, passed: dict | None) -> dict:
    """{"paused_after", "stale": [{phase, reason}]} for the banner."""
    if passed is None:
        return {"paused_after": (_json(out_dir / _STATE_FILE) or {}).get("paused_after"), "stale": []}
    stale = [
        {"phase": row["phase"], "reason": row.get("reason", "")}
        for row in passed.get("phases", []) if row["state"] == _STALE
    ]
    return {"paused_after": passed.get("paused_after"), "stale": stale}


def _decision_dict(decision: Decision) -> dict:
    return {
        "kind": decision.kind.value, "target": decision.target, "verdict": decision.verdict.value,
        "fix": decision.fix, "by": decision.by, "at": decision.at, "note": decision.note,
    }


def _reviewed(row: dict, kind: Kind, target: dict, decisions: Decisions) -> dict:
    """`row` with the kind and target of its decision, and the decision already recorded for it (or None)."""
    found = decisions.find(kind, **target)
    return {**row, "kind": kind.value, "target": target, "decision": _decision_dict(found) if found else None}


def _meta(settings: Settings, manifest: dict, semantic: dict) -> dict:
    stages = manifest.get("stages", {})
    unstructured = stages.get("unstructured", {})
    return {
        "database": settings.path("database.path").name,
        "now": (semantic.get("now") or {}).get("value"),
        "document_window": (unstructured.get("extra") or {}).get("coverage"),
        "stages": {
            name: {"built_at": stage.get("built_at"), "models": stage.get("models", {})}
            for name, stage in stages.items()
        },
    }


def _counts(roles, entities, documents, events, moves, drivers, review, edges) -> dict:
    columns = [column for table in roles["tables"].values() for column in table["columns"].values()]
    by_type = defaultdict(int)
    for entity in entities:
        by_type[entity["type"]] += 1
    return {
        "tables": len(roles["tables"]),
        "columns": len(columns),
        "columns_flagged": sum(bool(column.get("needs_review")) for column in columns),
        "entities": len(entities),
        "entities_by_type": dict(sorted(by_type.items())),
        "documents": len(documents),
        "events": len(events),
        "edges": len(edges),
        "moves": len(moves),
        "moves_with_drivers": len({row["move_id"] for row in drivers}),
        "candidate_drivers": len(drivers),
        "review_items": sum(line.startswith("- ") for line in review.splitlines()),
    }


def _tables(roles: dict, semantic: dict, decisions: Decisions) -> list[dict]:
    tables = []
    for name, table in roles["tables"].items():
        spec = semantic["tables"].get(name, {})
        tables.append({
            "name": name,
            "kind": table.get("kind"),
            "view": spec.get("view"),
            "columns": [
                _reviewed({
                    "column": column, "role": item.get("role"), "logical": item.get("logical_name"),
                    "description": item.get("description", ""), "confidence": item.get("confidence"),
                    "source": item.get("source"), "flagged": bool(item.get("needs_review")),
                }, Kind.COLUMN, {"table": name, "column": column}, decisions)
                for column, item in table["columns"].items()
            ],
        })
    return tables


def _rules(semantic: dict, decisions: Decisions) -> list[dict]:
    rules = [{"rule": "Data 'now' (latest actual date)", "value": (semantic.get("now") or {}).get("value")}]
    for name, spec in semantic["tables"].items():
        if spec.get("kind") != "measures":
            continue
        values = {
            "main measure": spec.get("primary_measure"),
            "never aggregate across": ", ".join(spec.get("denominations", [])) or "-",
            "scenario labels": (spec.get("scenario") or {}).get("values"),
            "time grains": ", ".join((spec.get("time") or {}).get("grains", [])),
        }
        rules += [
            _reviewed({"rule": f"{name}: {label}", "value": value}, Kind.RULE, {"table": name, "field": _RULE_FIELDS[label]}, decisions)
            for label, value in values.items()
        ]
        rules.append({"rule": f"{name}: hierarchy (child → parent)", "value": " → ".join(spec.get("hierarchy", [])) or "-"})
    rules += [{"rule": "Conversions (from context.md)", "value": semantic.get("conversions") or "none"}]
    rules += [{"rule": f"Synonyms of {key}", "value": ", ".join(value)} for key, value in (semantic.get("synonyms") or {}).items()]
    return rules


def _type_row(item: dict, decisions: Decisions) -> dict:
    if item["name"] in _ENTITY_TYPES_SKIPPED:
        return item
    return _reviewed(item, Kind.ENTITY_TYPE, {"name": item["name"]}, decisions)


def _entity(item: dict) -> dict:
    attributes = item.get("attributes", {})
    return {
        "id": item["id"], "type": item["type"], "name": item["canonical_name"], "aliases": item.get("aliases", []),
        "parents": attributes.get("parents", []), "part_of": attributes.get("part_of"),
        "tables": sorted({source["table"] for source in attributes.get("sources", [])}),
        "source": item.get("source"), "confidence": item.get("confidence"), "flagged": bool(item.get("needs_review")),
    }


def _name_matches(out_dir: Path, names: dict, decisions: Decisions) -> list[dict]:
    """Names matched with less than full certainty (or not at all), least sure first; one row per text."""
    found: dict[str, dict] = {}

    def note(text, entity_id, score, doc):
        if not text or (entity_id is not None and (score or 0) >= 1.0):
            return
        row = found.setdefault(text, {"text": text, "entity_id": entity_id, "score": score or 0, "docs": []})
        if doc and doc not in row["docs"]:
            row["docs"].append(doc)

    for edge in _jsonl(out_dir / "docs/tag_edges.jsonl"):
        note(edge.get("text"), edge["dst"], edge.get("confidence"), edge.get("evidence_doc"))
    for row in _jsonl(out_dir / "docs/resolved.jsonl"):
        for mention in row["mentions"]:
            note(mention["name"], mention["entity_id"], mention.get("score"), row["doc_id"])
        for event in row["events"]:
            for affect in event["affects"]:
                note(affect["entity"], affect.get("entity_id"), affect.get("match_score"), row["doc_id"])

    least_sure = sorted(found.values(), key=lambda row: (row["score"], row["text"]))[:_MAX_NAME_MATCHES]
    return [
        _reviewed({**row, "entity": names.get(row["entity_id"], row["entity_id"])}, Kind.NAME_MATCH, {"text": row["text"]}, decisions)
        for row in least_sure
    ]


def _moves(
    moves: list[dict], drivers: list[dict], by_id: dict, window: list[str] | None, keys_of: dict, decisions: Decisions,
) -> list[dict]:
    """Moves inside the news period first, then the rest; biggest change first in each group."""
    by_move = defaultdict(list)
    for row in drivers:
        by_move[row["move_id"]].append(row)

    def inside(move: dict) -> bool:
        return bool(window) and window[0] <= move["date"] <= window[1]

    biggest = sorted(moves, key=lambda move: -abs(move["change_pct"]))
    shown = [move for move in biggest if inside(move)][:_MAX_MOVES_INSIDE]
    shown += [move for move in biggest if not inside(move)][:_MAX_MOVES_OUTSIDE]
    result = []
    for move in shown:
        ranked = sorted(by_move.get(move["id"], []), key=lambda row: -row["score"])
        result.append(_reviewed({
            "id": move["id"], "entity": move["entity_id"], "date": move["date"], "direction": move["direction"],
            "change_pct": round(move["change_pct"], 2), "z": round(move.get("z") or 0, 1),
            "frequency": move.get("frequency"), "denomination": move.get("denomination"), "in_window": inside(move),
            "drivers": [_driver(row, by_id.get(row["event_id"], {}), move["id"], keys_of, decisions) for row in ranked],
        }, Kind.MOVE, {"id": move["id"]}, decisions))
    return result


def _driver(row: dict, event: dict, move_id: str, keys_of: dict, decisions: Decisions) -> dict:
    props = event.get("props", {})
    shown = {
        "event": event.get("name", row["event_id"]), "event_date": props.get("date"),
        "status": props.get("status"), "date_source": props.get("date_source"), "type": props.get("type"),
        "aspect": row.get("aspect"), "direction": row.get("direction"), "lag_days": row["lag_days"],
        "score": round(row["score"], 3), "relation": row["relation"], "doc": row.get("evidence_doc"),
        "quote": (row.get("evidence_quote") or "")[:_QUOTE_CHARS],
    }
    keys = keys_of.get(row["event_id"], [])
    if not keys:
        return shown
    return _reviewed(shown, Kind.DRIVER, {"move_id": move_id, "key": keys[0]}, decisions)


def _event(node: dict, keys: list[str], decisions: Decisions) -> dict:
    props = node["props"]
    shown = {
        "summary": node["name"], "type": props.get("type"), "date": props.get("date"),
        "status": props.get("status"), "date_source": props.get("date_source"),
        "entities": props.get("entity_ids", []), "docs": props.get("doc_ids", []),
        "affects": [
            {"entity": a.get("entity_id"), "aspect": a.get("aspect"), "direction": a.get("direction")}
            for a in props.get("affects", [])
        ],
        "flagged": bool(props.get("needs_review")),
    }
    if not keys:
        return shown
    return _reviewed(shown, Kind.EVENT, {"key": keys[0]}, decisions)


def _merged_events(events: list[dict], keys_of: dict, decisions: Decisions) -> list[dict]:
    """One row per pair (first extraction, another extraction) of an event merged from several articles."""
    rows = []
    for node in events:
        keys = keys_of.get(node["id"], [])
        for other in keys[1:]:
            row = {"summary": node["name"], "date": node["props"].get("date"), "pair": [keys[0], other]}
            rows.append(_reviewed(row, Kind.SAME_EVENT, {"keys": [keys[0], other]}, decisions))
    return rows


def _neighbours(entities: list[dict], edges: list[dict], by_id: dict) -> dict:
    """{entity id: [connection]} for catalog entities; sentence-backed links first, then tag, then data links, capped."""
    wanted = {item["id"] for item in entities if item["type"] not in _ENTITY_TYPES_SKIPPED}
    found = defaultdict(list)
    for edge in edges:
        for here, there, way in ((edge["src"], edge["dst"], "out"), (edge["dst"], edge["src"], "in")):
            if here in wanted:
                found[here].append(_connection(edge, way, by_id.get(there, {}), there))

    # Sorted before the cap, so the sentence-backed links are the ones kept.
    for items in found.values():
        items.sort(key=_rank)
    return {here: items[:_MAX_NEIGHBOURS] for here, items in found.items()}


def _rank(connection: dict) -> tuple[int, float]:
    if connection["evidence"]:
        tier = _SENTENCE_TIER
    elif connection["rel"] == _TAGGED_WITH:
        tier = _TAG_TIER
    else:
        tier = _DATA_TIER
    return tier, -(connection["confidence"] or 0)


def _connection(edge: dict, way: str, other: dict, other_id: str) -> dict:
    evidence = edge.get("evidence") or [
        {"doc": edge.get("evidence_doc"), "quote": edge.get("evidence_quote") or "", "confidence": edge.get("confidence")}
    ]
    sentences = [item for item in evidence if item.get("quote")]
    return {
        "rel": edge["rel"], "way": way, "other": other.get("name", other_id), "other_type": other.get("type", ""),
        "confidence": edge.get("confidence"), "flagged": bool(edge.get("needs_review")),
        "support": edge.get("support") or len(sentences),
        "evidence": [{"doc": item.get("doc"), "quote": item["quote"][:_QUOTE_CHARS]} for item in sentences[:_MAX_EVIDENCE]],
        "basis": "" if sentences else _basis(edge, evidence),
    }


def _basis(edge: dict, evidence: list[dict]) -> str:
    """Plain-language reason for a connection that has no sentence behind it."""
    rel = edge["rel"]
    if rel == _TAGGED_WITH:
        column = next((item["column"] for item in evidence if item.get("column")), None) or (edge.get("props") or {}).get("column")
        return f"from tag column {column}" if column else "from a tag column"
    if rel == _MENTIONS:
        return "name found in text"
    return "from the data" if rel in _FROM_THE_DATA else ""


def _yaml(path: Path):
    return read_yaml(path) if path.is_file() else None


def _json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


# ------------------------------------------------------------------ template

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Context Explorer</title>
<style>
:root { --bg:#f6f2ea; --paper:#fffdf8; --fg:#1b2230; --muted:#6c6a63; --line:#e4ddcf; --rule:#1b2230;
        --ink:#14213d; --ink-fg:#f6f2ea; --accent:#0f6e6e; --accent-soft:#e2efec;
        --up:#1f7a4d; --down:#b0413e; --flag:#b7791f; --chip:#efe9dd; --banner:#fbf0d4;
        --serif:"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
        --sans:"Avenir Next","Avenir","Segoe UI","Helvetica Neue",sans-serif;
        --mono:"SF Mono",Menlo,Consolas,"Liberation Mono",monospace; }
@media (prefers-color-scheme: dark) { :root { --bg:#12161f; --paper:#181d28; --fg:#e9e4d8; --muted:#9a968c; --line:#2a3140;
        --rule:#e9e4d8; --ink:#0b0f17; --ink-fg:#e9e4d8; --accent:#5cc2b8; --accent-soft:#16312f;
        --up:#5cc68f; --down:#e3807c; --flag:#e0b25a; --chip:#232a37; --banner:#33290f; } }
* { box-sizing:border-box; }
body { margin:0; color:var(--fg); font:15px/1.55 var(--sans);
       background:var(--bg) radial-gradient(circle at 1px 1px, rgba(20,33,61,.06) 1px, transparent 0) 0 0/22px 22px; }
code, .mono { font-family:var(--mono); font-size:.88em; }
header { position:sticky; top:0; z-index:3; background:var(--ink); color:var(--ink-fg); padding:14px 22px 12px;
         box-shadow:0 2px 0 var(--accent), 0 8px 24px rgba(0,0,0,.18); }
header h1 { margin:0; font:600 26px/1.1 var(--serif); letter-spacing:.2px; }
header h1 small { font:500 11px var(--sans); letter-spacing:2.5px; text-transform:uppercase; opacity:.7; margin-left:10px; }
#sub { color:var(--ink-fg); opacity:.72; font-size:13px; margin-top:4px; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
#search { flex:1 1 320px; max-width:520px; padding:8px 12px; border:1px solid transparent; border-radius:999px;
          background:rgba(255,255,255,.1); color:var(--ink-fg); }
#search::placeholder, .bar input::placeholder { color:var(--ink-fg); opacity:.6; }
.bar input { padding:7px 12px; border:1px solid rgba(255,255,255,.18); border-radius:999px; background:transparent; color:var(--ink-fg); }
#export { padding:7px 16px; border-radius:999px; border:0; background:var(--accent); color:#fff; font-weight:600; letter-spacing:.3px; }
#export:hover { filter:brightness(1.1); }
#drafts { color:var(--ink-fg); opacity:.75; }
.need { border-color:var(--down) !important; box-shadow:0 0 0 2px var(--down); }
main { max-width:1180px; margin:0 auto; padding:22px 22px 60px; }
.steps { background:var(--paper); border:1px solid var(--line); border-top:4px solid var(--accent); border-radius:10px;
         padding:16px 18px 18px; margin:0 0 18px; box-shadow:0 1px 0 var(--line); animation:rise .5s ease both; }
.steps h2 { margin:0 0 4px; font:600 19px var(--serif); }
.steps > p { margin:0 0 12px; color:var(--muted); font-size:13.5px; }
.steps ol { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; counter-reset:step; }
.steps li { counter-increment:step; position:relative; padding:12px 12px 12px 54px; background:var(--bg); border:1px solid var(--line); border-radius:8px; font-size:14px; }
.steps li::before { content:counter(step); position:absolute; left:12px; top:8px; font:600 30px/1 var(--serif); color:var(--accent); }
.steps li > b:first-child { display:block; font-family:var(--serif); font-size:15px; margin-bottom:2px; }
.steps code { user-select:all; background:var(--chip); padding:1px 6px; border-radius:4px; }
.steps .note { margin:12px 0 0; font-size:13px; color:var(--muted); }
details { background:var(--paper); border:1px solid var(--line); border-radius:10px; margin:14px 0;
          box-shadow:0 1px 0 var(--line); animation:rise .45s ease both; }
main > details:nth-of-type(2) { animation-delay:.04s } main > details:nth-of-type(3) { animation-delay:.08s }
main > details:nth-of-type(4) { animation-delay:.12s } main > details:nth-of-type(5) { animation-delay:.16s }
main > details:nth-of-type(n+6) { animation-delay:.2s }
@keyframes rise { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
summary { cursor:pointer; padding:13px 18px; font:600 17px var(--serif); list-style:none; display:flex; align-items:baseline; gap:8px; }
summary::-webkit-details-marker { display:none; }
summary::before { content:"›"; font:600 18px var(--sans); color:var(--accent); transition:transform .2s; display:inline-block; width:12px; }
details[open] > summary::before { transform:rotate(90deg); }
summary:hover { color:var(--accent); }
.body { padding:0 18px 18px; overflow-x:auto; }
h3 { font:600 16px var(--serif); margin:16px 0 8px; } h4 { font:600 14px var(--serif); margin:14px 0 6px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(165px,1fr)); gap:10px; margin:6px 0 12px; }
.stat { border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--bg); }
.stat b { display:block; font:600 28px/1.1 var(--serif); color:var(--ink); }
@media (prefers-color-scheme: dark) { .stat b { color:var(--fg); } }
.stat span { color:var(--muted); font-size:12.5px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font:600 11px var(--sans); letter-spacing:1px; text-transform:uppercase; border-bottom:2px solid var(--rule); }
tbody tr:hover { background:var(--accent-soft); }
.chip { display:inline-block; padding:1px 8px; border-radius:10px; background:var(--chip); font-size:12px; margin:1px 2px; }
.flag { color:var(--flag); font-weight:600; }
.up { color:var(--up); font-weight:600; } .down { color:var(--down); font-weight:600; }
.muted { color:var(--muted); }
.small { font-size:12.5px; font-weight:400; margin-top:2px; }
.term { color:var(--muted); font:400 12px var(--sans); letter-spacing:.3px; }
.intro { margin:2px 0 14px; color:var(--fg); max-width:820px; }
.quote { border-left:3px solid var(--accent); padding-left:10px; color:var(--muted); font-family:var(--serif); font-style:italic; min-width:240px; margin-bottom:4px; }
.move { border:1px solid var(--line); border-radius:8px; margin:8px 0; background:var(--bg); box-shadow:none; animation:none; }
.move > summary { font:500 14.5px var(--sans); }
select { padding:6px; border-radius:6px; border:1px solid var(--line); background:var(--paper); color:var(--fg); max-width:100%; }
pre { white-space:pre-wrap; font:12.5px/1.5 var(--mono); }
.hidden { display:none; }
.banner { background:var(--banner); border:1px solid var(--flag); border-left-width:5px; border-radius:8px; padding:10px 14px; margin:0 0 14px; }
button { padding:3px 10px; border:1px solid var(--line); border-radius:6px; background:var(--paper); color:var(--fg); cursor:pointer; font:inherit; font-size:13px; transition:border-color .15s, background .15s; }
button:hover { border-color:var(--accent); background:var(--accent-soft); }
.rv { white-space:nowrap; }
.rv .form { white-space:normal; min-width:220px; margin-top:6px; }
.rv .form label { display:block; font-size:12.5px; color:var(--muted); margin-top:4px; }
.rv .form input, .rv .form select { width:100%; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:var(--paper); color:var(--fg); }
.badge { display:inline-block; font-size:12px; padding:1px 8px; border-radius:10px; background:var(--chip); margin-right:4px; white-space:normal; }
.badge.correct { color:var(--up); } .badge.wrong { color:var(--down); } .badge.fix { color:var(--flag); }
.rvbar { margin:8px 0; }
a[data-act] { cursor:pointer; color:var(--accent); font-size:12px; }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
</head>
<body>
<header>
  <h1>Context Explorer <small>context layer · review</small></h1>
  <div class="muted" id="sub"></div>
  <div class="bar">
    <input id="search" placeholder="Search names, columns, events, price changes…" autocomplete="off">
    <input id="reviewer" placeholder="Your name (for review marks)" autocomplete="off">
    <button id="export" type="button">Export decisions</button>
    <span class="muted" id="drafts"></span>
  </div>
</header>
<main id="app"></main>
<datalist id="ents"></datalist>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);

// PURE-BEGIN (no page access; the tests run this block under node)
function identity(kind, target) {
  const parts = Object.keys(target).sort().map(k =>
    [k, kind === 'same_event' && k === 'keys' ? target[k].map(String).sort() : target[k]]);
  return JSON.stringify([kind, parts]);
}
// Union of both lists; per (kind, target) the newest `at` wins, ties go to the later entry.
function mergeDecisions(existing, added) {
  const all = [...existing, ...added], winner = new Map();
  all.forEach((d, i) => {
    const id = identity(d.kind, d.target), at = String(d.at || ''), cur = winner.get(id);
    if (!cur || at >= cur.at) winner.set(id, {at, index: i});
  });
  return all.filter((d, i) => winner.get(identity(d.kind, d.target)).index === i);
}
// Drafts the page already embeds (same verdict, fix and `at`) or that are older than its decision are stale: drop them.
function liveDrafts(existing, drafts) {
  const embedded = new Map(existing.map(d => [identity(d.kind, d.target), d]));
  return drafts.filter(d => {
    const e = embedded.get(identity(d.kind, d.target));
    if (!e) return true;
    const same = d.verdict === e.verdict && JSON.stringify(d.fix ?? {}) === JSON.stringify(e.fix ?? {}) && d.at === e.at;
    return !same && String(d.at || '') >= String(e.at || '');
  });
}
// One field per line, each value as JSON (which is valid YAML).
function toYaml(decisions) {
  const field = (key, value) => `${key}: ${JSON.stringify(value)}`;
  const items = decisions.map(d => [
    `  - ${field('kind', d.kind)}`,
    ...['target', 'verdict', 'fix', 'by', 'at', 'note'].map(k => `    ${field(k, d[k] ?? (k === 'fix' ? {} : ''))}`),
  ].join('\n'));
  return 'decisions:' + (items.length ? '\n' + items.join('\n') : ' []') + '\n';
}
// PURE-END

const app = document.getElementById('app');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const dir = d => d === 'up' ? '<span class="up">▲ up</span>' : d === 'down' ? '<span class="down">▼ down</span>' : `<span class="muted">${esc(d || 'unknown')}</span>`;
const flag = f => f ? ' <span class="flag" title="needs review">⚑</span>' : '';
const chips = xs => (xs || []).map(x => `<span class="chip">${esc(x)}</span>`).join('');
const section = (title, html, open) => `<details ${open ? 'open' : ''}><summary>${title}</summary><div class="body">${html}</div></details>`;
const table = (head, rows) => `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
const fmt = v => typeof v === 'object' && v !== null ? esc(JSON.stringify(v)) : esc(v);

// ---- review marks: drafts live in localStorage (inside try/catch) until exported
const STORE = 'ctx-review:' + D.meta.database;
let saved = {reviewer: '', session: []};
try {
  const raw = JSON.parse(localStorage.getItem(STORE) || 'null');
  if (raw && Array.isArray(raw.session)) saved = {reviewer: String(raw.reviewer || ''), session: liveDrafts(D.decisions, raw.session)};
} catch (e) { /* storage blocked or damaged: start with no drafts */ }
const persist = () => { try { localStorage.setItem(STORE, JSON.stringify(saved)); } catch (e) { /* storage blocked */ } };

// What each kind of row offers. `fix` lists the form fields; ok / no say whether ✓ / ✗ make sense.
const F = {
  role: {key: 'role', label: 'Role', input: 'role'},
  logical: {key: 'logical_name', label: 'Readable name', input: 'text'},
  description: {key: 'description', label: 'Description', input: 'text'},
  name: {key: 'name', label: 'New name', input: 'text'},
  canonical: {key: 'canonical_name', label: 'Name', input: 'text'},
  aliases: {key: 'add_aliases', label: 'Add other names (comma separated)', input: 'list'},
  mergeEntity: {key: 'merge_into', label: 'Same as (pick one)', input: 'entity'},
  mergeType: {key: 'merge_into', label: 'Merge into kind', input: 'eventType'},
  entityId: {key: 'entity_id', label: 'Always match to (pick one)', input: 'entity'},
  type: {key: 'type', label: 'Kind of event', input: 'eventType'},
  date: {key: 'date', label: 'Date (YYYY-MM-DD)', input: 'text'},
  status: {key: 'status', label: 'Has it happened?', input: 'status'},
  affects: {key: 'affects', label: 'What it affects', input: 'affects'},
  value: {key: 'value', label: 'New value (text, or JSON for a list)', input: 'json'},
};
const KINDS = {
  column: {ok: 1, no: 0, fix: [F.role, F.logical, F.description]},
  entity_type: {ok: 1, no: 1, fix: [F.name, F.description]},
  event_type: {ok: 1, no: 1, fix: [F.name, F.description, F.mergeType]},
  entity: {ok: 1, no: 1, fix: [F.canonical, F.aliases, F.mergeEntity]},
  new_entity: {ok: 1, no: 1, fix: [F.mergeEntity]},
  name_match: {ok: 0, no: 1, fix: [F.entityId]},
  event: {ok: 1, no: 1, fix: [F.type, F.date, F.status, F.affects]},
  same_event: {ok: 1, no: 1, fix: []},
  move: {ok: 1, no: 1, fix: []},
  driver: {ok: 1, no: 1, fix: []},
  rule: {ok: 0, no: 0, fix: [F.value]},
};
const STATUSES = ['reported', 'announced', 'forecast'], DIRECTIONS = ['up', 'down', 'unknown'];
const eventTypes = D.ontology.event_types.map(e => e.name);

const ROWS = [];
const rv = row => { if (!row.kind) return ''; ROWS.push(row); return `<span class="rv" data-r="${ROWS.length - 1}"></span>`; };
const rvCell = row => `<td>${rv(row)}</td>`;

let effectiveCache = null;
const effective = () => effectiveCache ??= new Map(mergeDecisions(D.decisions, saved.session).map(d => [identity(d.kind, d.target), d]));
const isDraft = d => saved.session.includes(d);
const reviewer = document.getElementById('reviewer');
reviewer.value = saved.reviewer;
const today = () => { const t = new Date(), p = n => String(n).padStart(2, '0'); return `${t.getFullYear()}-${p(t.getMonth() + 1)}-${p(t.getDate())}`; };
const fixText = fix => Object.entries(fix || {}).map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ');
const VERDICT = {correct: '✓ correct', wrong: '✗ wrong', fix: 'fix'};

function drawMark(holder) {
  const row = ROWS[+holder.dataset.r], cfg = KINDS[row.kind];
  const cur = effective().get(identity(row.kind, row.target));
  const btn = (act, label, title) => `<button type="button" data-act="${act}" title="${title}">${label}</button>`;
  holder.innerHTML =
    (cur ? `<span class="badge ${esc(cur.verdict)}" title="${esc(cur.note)}">${esc(VERDICT[cur.verdict] || cur.verdict)}${cur.verdict === 'fix' ? ' ' + esc(fixText(cur.fix)) : ''} · ${esc(cur.by)} · ${esc(cur.at)}${cur.note ? ' · “' + esc(cur.note) + '”' : ''}</span>` +
      (isDraft(cur) ? '<a data-act="undo">undo</a> ' : '') : '') +
    (cfg.ok ? btn('correct', '✓', 'looks right') : '') + (cfg.no && !(row.kind === 'name_match' && !row.entity_id) ? btn('wrong', '✗', 'wrong') : '') +
    (cfg.fix.length ? btn('fix', 'fix', 'correct it') : '');
}
const drawAll = () => { effectiveCache = null; document.querySelectorAll('.rv').forEach(drawMark); updateDrafts(); };
function updateDrafts() {
  const n = saved.session.length;
  document.getElementById('drafts').textContent = n ? `${n} mark${n > 1 ? 's' : ''} made in this browser, not yet exported` : '';
}

function record(row, verdict, fix, note) {
  const by = reviewer.value.trim();
  if (!by) { reviewer.classList.add('need'); reviewer.focus(); return false; }
  const id = identity(row.kind, row.target);
  saved.session = saved.session.filter(d => identity(d.kind, d.target) !== id);
  saved.session.push({kind: row.kind, target: row.target, verdict, fix, by, at: today(), note: note || ''});
  persist();
  drawAll();
  return true;
}
reviewer.addEventListener('input', () => { reviewer.classList.remove('need'); saved.reviewer = reviewer.value; persist(); });

const select = (key, options, value) => `<select data-f="${key}"><option value=""></option>${options.map(o => `<option ${o === value ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
function fieldHtml(f, old) {
  const value = old?.[f.key];
  const control = {
    role: () => select(f.key, D.roles, value),
    eventType: () => select(f.key, eventTypes, value),
    status: () => select(f.key, STATUSES, value),
    entity: () => `<input data-f="${f.key}" list="ents" value="${esc(value)}">`,
    list: () => `<input data-f="${f.key}" value="${esc((value || []).join(', '))}">`,
    json: () => `<input data-f="${f.key}" value="${esc(typeof value === 'string' ? value : value === undefined ? '' : JSON.stringify(value))}">`,
    affects: () => `<input data-f="affects.entity" list="ents" placeholder="thing affected"> ${select('affects.direction', DIRECTIONS)}<input data-f="affects.aspect" placeholder="what changes (e.g. price)">`,
    text: () => `<input data-f="${f.key}" value="${esc(value)}">`,
  }[f.input]();
  return `<label>${f.label}${control}</label>`;
}
function openForm(holder, row) {
  const cur = effective().get(identity(row.kind, row.target));
  const old = cur && cur.verdict === 'fix' ? cur.fix : {};
  holder.innerHTML = `<div class="form">${KINDS[row.kind].fix.map(f => fieldHtml(f, old)).join('')}
    <label>Note (optional)<input data-f="note" value="${esc(cur?.note)}"></label>
    <div style="margin-top:6px"><button type="button" data-act="save">Save</button> <button type="button" data-act="cancel">Cancel</button></div></div>`;
}
function readForm(holder, row) {
  const get = k => holder.querySelector(`[data-f="${k}"]`)?.value.trim() || '';
  const fix = {};
  for (const f of KINDS[row.kind].fix) {
    if (f.input === 'affects') {
      if (get('affects.entity')) fix.affects = [{entity: get('affects.entity'), ...(get('affects.aspect') && {aspect: get('affects.aspect')}), ...(get('affects.direction') && {direction: get('affects.direction')})}];
      continue;
    }
    const v = get(f.key);
    if (!v) continue;
    if (f.input === 'list') fix[f.key] = v.split(',').map(x => x.trim()).filter(Boolean);
    else if (f.input === 'json' && /^[\[{]/.test(v)) { try { fix[f.key] = JSON.parse(v); } catch (e) { fix[f.key] = v; } }
    else fix[f.key] = v;
  }
  return {fix, note: get('note')};
}
app.addEventListener('click', ev => {
  const control = ev.target.closest('[data-act]');
  const holder = control && control.closest('.rv');
  if (!holder) return;
  const row = ROWS[+holder.dataset.r], act = control.dataset.act;
  if (act === 'correct') record(row, 'correct', {});
  else if (act === 'wrong') record(row, 'wrong', row.kind === 'name_match' ? {not_entity_id: row.entity_id} : {});
  else if (act === 'fix') openForm(holder, row);
  else if (act === 'cancel') drawMark(holder);
  else if (act === 'undo') {
    const id = identity(row.kind, row.target);
    saved.session = saved.session.filter(d => identity(d.kind, d.target) !== id);
    persist();
    drawAll();
  } else if (act === 'save') {
    const {fix, note} = readForm(holder, row);
    if (Object.keys(fix).length) record(row, 'fix', fix, note);
  }
});
document.getElementById('export').addEventListener('click', () => {
  const text = toYaml(mergeDecisions(D.decisions, saved.session));
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], {type: 'text/yaml'}));
  link.download = 'decisions.yaml';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

const m = D.meta, c = D.counts;
document.getElementById('sub').textContent =
  `${m.database} · latest date in the numbers: ${m.now ?? '-'} · news articles cover: ${(m.document_window || []).join(' to ') || '-'}`;
// Only catalog entities: a name match or merge can point at those, not at a tag's new entity.
document.getElementById('ents').innerHTML = D.entities.filter(e => e.kind === 'entity').map(e => `<option value="${esc(e.id)}">${esc(e.name)}</option>`).join('');

const W = m.document_window || [];
const period = W.length ? `${W[0]} to ${W[1]}` : 'none';
const intro = t => `<p class="intro">${t}</p>`;
const SOURCE = {inferred: 'worked out by the code', metadata: 'read from the data', 'context.md': 'from context.md', human: 'confirmed by a person'};
const src = v => esc(SOURCE[v] || v || '-');
const STATUS = {reported: 'already happened', announced: 'announced', forecast: 'expected (forecast)'};
const PREV = {weekly: 'previous week', monthly: 'previous month', daily: 'previous day', quarterly: 'previous quarter', yearly: 'previous year'};

const S = D.state || {};
let html = `<section class="steps" aria-label="How to apply a review">
  <h2>How to apply a review</h2>
  <p>Fix what is wrong right here on the page, then hand your decisions back to the build.</p>
  <ol>
    <li><b>Fix by clicking</b>Find the row to fix and click <b>✓</b> (correct), <b>✗</b> (wrong) or <b>fix</b> (change it). Enter your name at the top first.</li>
    <li><b>Export decisions</b>When your fixes are done, click <b>Export decisions</b> at the top. A <code>decisions.yaml</code> file downloads.</li>
    <li><b>Move the file</b>Put the downloaded file in the project's <code>review</code> folder, as <code>review/decisions.yaml</code> (replace the old one).</li>
    <li><b>Apply</b>In the project folder run <code>ctx review apply</code>. It updates the context and this page.</li>
  </ol>
  <p class="note">Your clicks are kept in this browser until you export, so reloading the page is safe. Fixes to <b>column roles</b>, <b>entity types</b> or <b>new event types</b> are saved but take effect only after the command <code>ctx review apply</code> prints (a <code>ctx build --from &lt;phase&gt;</code>).</p>
</section>`;
if (S.paused_after) html += `<div class="banner">The build is paused after <b>${esc(S.paused_after)}</b>. Check the rows below, export your decisions into <code>review/decisions.yaml</code>, then run <code>ctx build --continue</code>.</div>`;
if ((S.stale || []).length) html += `<div class="banner">Out of date: ${S.stale.map(s => `<b>${esc(s.phase)}</b>${s.reason ? ' (' + esc(s.reason) + ')' : ''}`).join(', ')}. Run <code>ctx build</code> to bring them up to date.</div>`;

const stats = [['Tables', c.tables], ['Columns understood', c.columns], ['Columns to double-check', c.columns_flagged],
  ['Things recognised (people, places, products…)', c.entities], ['News articles', c.documents], ['Events found in the news', c.events],
  ['Connections between things', c.edges], ['Unusual price changes', c.moves], ['Price changes with supporting news', c.moves_with_drivers],
  ['Items to double-check', c.review_items]];
const models = Object.entries(m.stages).map(([s, v]) =>
  `<tr><td>${esc(s)}</td><td>${esc(v.built_at)}</td><td>${fmt(v.models)}</td></tr>`);
html += section('Overview <span class="term">(build summary)</span>',
  intro(`What the code built from <b>${esc(m.database)}</b>, with no knowledge of this dataset written into the code. The numbers' latest date ("now") is <b>${esc(m.now ?? '-')}</b>; the news articles cover <b>${esc(period)}</b>.`) +
  `<div class="grid">${stats.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('')}</div>
   <p class="muted">Things recognised, by kind: ${Object.entries(c.entities_by_type).map(([t, n]) => `<span class="chip">${esc(t)}: ${n}</span>`).join('')}</p>
   ${table(['Build step', 'Built at', 'AI models used'], models)}`, true);

html += section('What each column means <span class="term">(column roles)</span>', intro('Every column in your data and what the code decided it is: a date, a price, a product name, part of an article, and so on. <span class="flag">⚑</span> marks a guess the code was not sure about. A changed role takes effect with <code>ctx build --from roles</code>.') +
  D.tables.map(t =>
  `<h3>${esc(t.name)} <span class="chip">${t.kind === 'text' ? 'articles' : t.kind === 'measures' ? 'numbers over time' : esc(t.kind)}</span></h3>` +
  table(['Column', 'What it is', 'Readable name', 'Description', 'How sure (0–1)', 'Where it came from', 'Review'], t.columns.map(col =>
    `<tr class="row" data-q="${esc((col.column + ' ' + col.role + ' ' + col.logical + ' ' + col.description).toLowerCase())}">
     <td>${esc(col.column)}${flag(col.flagged)}</td><td><span class="chip">${esc((col.role || '').replace(/_/g, ' '))}</span></td><td>${esc(col.logical)}</td>
     <td>${esc(col.description)}</td><td>${col.confidence ?? ''}</td><td>${src(col.source)}</td>${rvCell(col)}</tr>`))).join(''));

html += section('Rules for using the numbers <span class="term">(business rules, semantic layer)</span>', intro('Rules the code follows so answers stay correct, for example which value is the main price, what must never be added together (different currencies or units), and how actual and forecast values are told apart.') +
  table(['Rule', 'Value', 'Review'], D.rules.map(r => `<tr><td>${esc(r.rule)}</td><td>${fmt(r.value)}</td>${rvCell(r)}</tr>`)));

const o = D.ontology;
html += section('Kinds of things and how they connect <span class="term">(ontology)</span>', intro('The kinds of things found in the data (products, places, articles, events…), how they relate to each other, and the kinds of events the news reports. Changes here take effect with <code>ctx build --from ontology</code> (new event types: <code>--from event-types</code>).') +
  table(['Kind of thing', 'Description', 'Review'], o.types.map(t => `<tr><td>${esc(t.name)}</td><td>${esc(t.description)}</td>${rvCell(t)}</tr>`)) +
  table(['From', 'Connection', 'To'], o.relations.map(r => `<tr><td>${esc(r.from)}</td><td><span class="chip">${esc(r.name.replace(/_/g, ' ').toLowerCase())}</span></td><td>${esc(r.to)}</td></tr>`)) +
  `<h4>Kinds of events in the news</h4>` + table(['Kind', 'Description', 'Review'], o.event_types.map(e => `<tr><td>${esc(e.name.replace(/_/g, ' '))}</td><td>${esc(e.description)}</td>${rvCell(e)}</tr>`)));

html += section(`Everything the code recognises (${D.entities.length}) <span class="term">(entity catalog)</span>`, intro('Each product, place or group the code found, with its other names (so "GP" and "Great Plains" count as the same place), what it belongs to, and which table it came from. Things marked <b>new from a tag</b> were not in the data tables; they came from an article tag and are guesses until checked.') +
  table(['Name', 'Kind', 'Other names', 'Belongs to', 'Found in', 'Where it came from', 'Review'],
  D.entities.map(e => `<tr class="row" data-q="${esc((e.name + ' ' + e.type + ' ' + e.aliases.join(' ') + ' ' + e.id).toLowerCase())}">
    <td>${esc(e.name)}${flag(e.flagged)}${e.kind === 'new_entity' ? ' <span class="chip">new from a tag</span>' : ''}<div class="muted">${esc(e.id)}</div></td><td>${esc(e.type)}</td><td>${chips(e.aliases)}</td>
    <td>${chips(e.parents)}${e.part_of ? chips([e.part_of]) : ''}</td><td>${chips(e.tables)}</td><td>${src(e.source)}</td>${rvCell(e)}</tr>`)));

html += section(`Names matched with some doubt (${D.name_matches.length}) <span class="term">(name matches)</span>`, intro('Words in articles that the code matched to a known thing without being fully sure, least sure first. <b>✗</b> means "never match this text to that thing"; <b>fix</b> picks the right one.') +
  table(['Text in the articles', 'Matched to', 'How sure (0–1)', 'Articles', 'Review'], D.name_matches.map(n =>
  `<tr class="row" data-q="${esc((n.text + ' ' + (n.entity || '')).toLowerCase())}"><td>${esc(n.text)}</td><td>${esc(n.entity || 'no match')}</td><td>${n.score}</td><td>${chips(n.docs.slice(0, 3))}</td>${rvCell(n)}</tr>`)));

const moveRow = mv => {
  const n = mv.drivers.length;
  const what = mv.direction === 'up' ? '<span class="up">▲ price up</span>' : mv.direction === 'down' ? '<span class="down">▼ price down</span>' : 'price changed';
  const empty = mv.in_window
    ? 'No supporting news articles were found in the dataset for this price change.'
    : `No news articles in the dataset cover this date (the news runs ${esc(period)}), so no supporting articles can exist.`;
  return `<details class="move row" data-q="${esc((mv.entity + ' ' + mv.date + ' ' + mv.drivers.map(d => d.event).join(' ')).toLowerCase())}">
   <summary>${esc(mv.entity)} · ${esc(mv.date)} · ${what} ${Math.abs(mv.change_pct)}% vs ${PREV[mv.frequency] || 'previous period'} · ${n ? `${n} supporting article${n > 1 ? 's' : ''}` : '<span class="muted">no supporting news</span>'}
   <div class="muted small">Unusual for this series: about ${Math.abs(mv.z)}× its normal swing</div></summary>
   <div class="body"><div class="rvbar">Keep this price change? ${rv(mv)}</div>${n ? table(['#', 'What happened', 'Has it happened?', 'What it says changes', 'Days before', 'Article and quote', 'Review'],
     mv.drivers.map((d, i) => `<tr><td>${i + 1}</td><td>${esc(d.event)}<div class="muted">${esc((d.type || '').replace(/_/g, ' '))} · ${esc(d.event_date)}${d.date_source === 'publication' ? ' (date the article was published)' : ''}</div><div class="muted small">match strength ${d.score}</div></td>
       <td><span class="chip">${esc(STATUS[d.status] || d.status || '-')}</span></td><td>${esc(d.aspect || '-')} ${dir(d.direction)}</td><td>${d.lag_days}</td>
       <td><div class="quote">${esc(d.quote) || '-'}</div><div class="muted">article ${esc(d.doc)}</div></td>${rvCell(d)}</tr>`))
     : `<p class="muted">${empty}</p>`}</div></details>`;
};
const during = D.moves.filter(mv => mv.in_window), outside = D.moves.filter(mv => !mv.in_window);
html += section('Price changes and the news behind them <span class="term">(moves and candidate drivers)</span>', intro(`Unusually large price changes, biggest change first. A change is listed when it is big for that series compared with its own recent history. Open one to see news articles from the days before it that may explain it, with the exact sentence. These are possible explanations to read, not proof of cause. On a supporting article, <b>✓</b> keeps it and <b>✗</b> drops it.`) +
  `<h3>During the news period (${esc(period)})</h3>` +
  (during.length ? during.map(moveRow).join('') : '<p class="muted">No unusual price changes during the news period.</p>') +
  `<h3>Before or after the news period</h3>` + intro('There are no news articles for these dates, so they can never have supporting articles.') +
  (outside.length ? outside.map(moveRow).join('') : '<p class="muted">None.</p>'));

html += section(`Events found in the news (${D.events.length}) <span class="term">(extracted events)</span>`, intro('Things that happened according to the articles (a drought, a strike, a new tariff…), with what each one says changes, and the articles it came from. Articles about the same event are merged into one.') +
  table(['Date', 'What happened', 'Kind', 'Has it happened?', 'What it says changes', 'Articles', 'Review'],
  D.events.map(ev => `<tr class="row" data-q="${esc((ev.summary + ' ' + ev.type + ' ' + ev.entities.join(' ')).toLowerCase())}">
    <td>${esc(ev.date)}${ev.date_source === 'publication' ? '<div class="muted">date the article was published</div>' : ''}</td><td>${esc(ev.summary)}${flag(ev.flagged)}</td>
    <td>${esc((ev.type || '').replace(/_/g, ' '))}</td><td><span class="chip">${esc(STATUS[ev.status] || ev.status || '-')}</span></td>
    <td>${ev.affects.map(a => `${esc(a.entity)} · ${esc(a.aspect || '-')} ${dir(a.direction)}`).join('<br>') || '-'}</td><td>${chips(ev.docs)}</td>${rvCell(ev)}</tr>`)));

html += section(`Events merged from several articles (${D.merged_events.length}) <span class="term">(same event)</span>`, intro('Articles the code decided describe one event. For each pair, <b>✓</b> means "yes, the same event" and <b>✗</b> means "keep them apart". An article is named by its id and the number of the event inside it.') +
  table(['Date', 'Merged event', 'The two articles', 'Review'], D.merged_events.map(p =>
  `<tr class="row" data-q="${esc((p.summary + ' ' + p.pair.join(' ')).toLowerCase())}"><td>${esc(p.date)}</td><td>${esc(p.summary)}</td><td>${chips(p.pair)}</td>${rvCell(p)}</tr>`)));

const ids = Object.keys(D.neighbours).sort();
const names = Object.fromEntries(D.entities.map(e => [e.id, e.name]));
html += section('Connections explorer <span class="term">(knowledge graph)</span>', intro('Pick anything the code recognises to see what it is connected to: articles about it, events that affect it, the group or place it belongs to, with the sentences behind each connection. Links backed by a sentence come first, then article tags, then links taken from the data. This view is read-only.') +
  `<select id="pick"><option value="">— choose one (${ids.length}) —</option>${ids.map(id => `<option value="${esc(id)}">${esc(names[id] || id)} · ${esc(id)}</option>`).join('')}</select><div id="nb"></div>`);

html += section('Needs a human check <span class="term">(review items, REVIEW.md)</span>', intro('Guesses the code was not sure about. They are kept in the files but not used to answer questions until someone checks them.') +
  (D.review ? `<pre>${esc(D.review)}</pre>` : '<p class="muted">Nothing flagged.</p>'));
app.innerHTML = html;
drawAll();

// Why a connection exists: up to three sentences with their articles, else the plain-language basis.
const why = n => n.evidence.length
  ? (n.support > 1 ? `<div class="muted small">${n.support} articles agree</div>` : '') +
    n.evidence.map(e => `<div class="quote">${esc(e.quote)}<div class="small">article ${esc(e.doc)}</div></div>`).join('')
  : `<span class="muted">${esc(n.basis) || '-'}</span>`;
document.getElementById('pick').addEventListener('change', ev => {
  const list = D.neighbours[ev.target.value] || [];
  document.getElementById('nb').innerHTML = list.length ? table(['Connection', 'Connected to', 'Kind', 'How sure (0–1)', 'Evidence'],
    list.map(n => `<tr><td>${n.way === 'out' ? '→' : '←'} <span class="chip">${esc(n.rel.replace(/_/g, ' ').toLowerCase())}</span>${flag(n.flagged)}</td><td>${esc(n.other)}</td>
      <td>${esc(n.other_type)}</td><td>${n.confidence ?? ''}</td><td>${why(n)}</td></tr>`)) : '<p class="muted">No connections.</p>';
});

document.getElementById('search').addEventListener('input', ev => {
  const q = ev.target.value.trim().toLowerCase();
  document.querySelectorAll('.row').forEach(r => r.classList.toggle('hidden', q !== '' && !r.dataset.q.includes(q)));
  if (q) document.querySelectorAll('details').forEach(d => { if (d.querySelector('.row:not(.hidden)')) d.open = true; });
});
</script>
</body>
</html>
"""
