# The context files: what each one holds and how to use it

The context layer is **plain files plus SQL views**. You don't need our agent or tools. Load the files into Python, SQL, a graph database, an LLM prompt or another agent framework.

The build writes to `artifacts/<database file stem>/`, for example `artifacts/example/`. The samples come from the synthetic example database (`examples/make_example_db.py`). Overview: [README](../README.md). Commands: [RUNNING.md](RUNNING.md).

Sections: [files](#files-for-using-the-context) · [recipes](#recipes) · [our tools](#our-tools-optional)

---

## Files for using the context

- **Meaning of the data:** [`semantic.yaml`](#semanticyaml) (how each table may be used), [`structured/views.sql`](#structuredviewssql) (read-only `sem_*` SQL views), [`roles.yaml`](#rolesyaml) (what every column is)
- **Things and names:** [`entities.yaml`](#entitiesyaml) (every product, place and group), [`graph/entities_text.yaml`](#graphentities_textyaml) (extra entities found only in article tags)
- **Kinds:** [`ontology.yaml`](#ontologyyaml) (kinds of things and relations), [`event_types.yaml`](#event_typesyaml) (kinds of events)
- **Price changes:** [`structured/moves.jsonl`](#structuredmovesjsonl)
- **Documents:** [`docs/documents.jsonl`](#docsdocumentsjsonl), [`docs/chunks.parquet`](#docschunksparquet) (pieces with embeddings, lists of numbers that capture meaning), [`docs/fts.duckdb`](#docsftsduckdb) (keyword index, BM25 being a standard keyword-ranking formula)
- **Events and the graph:** [`docs/events.jsonl`](#docseventsjsonl), [`graph/nodes.jsonl`, `graph/edges.jsonl`](#graphnodesjsonl-and-graphedgesjsonl)
- **News behind price changes:** [`links/candidate_drivers.jsonl`](#linkscandidate_driversjsonl), [`links/nodes.jsonl`, `links/edges.jsonl`](#linksnodesjsonl-and-linksedgesjsonl)
- **Review:** [`REVIEW.md`](#reviewmd), [`review/decisions.yaml`](#reviewdecisionsyaml)
- **Overview:** [`context_explorer.html`](#context_explorerhtml), one offline page showing all of the above

Every generated item carries a `source`: `metadata` (read from the data), `inferred` (worked out by the code or the LLM), `context.md` or `human`; and a `confidence`. Items below 0.8 are marked `needs_review: true`. The files keep them, but our tools ignore them until a person reviews them. You can do the same, or use them anyway.

### `semantic.yaml`

**Holds:** the semantic layer, the rules for using each table: entity and label columns, hierarchy, business date and its range, time grains, measures and how they combine, the main measure, the currency and unit columns you must never add up across, how to tell actual and projected rows apart, synonyms, unit conversions from `context.md`, and the data's "now" (its latest actual date).

```yaml
now: {value: '2025-06-30', source: inferred}
tables:
  grain_quotes:
    view: sem_grain_quotes
    kind: measures
    entity: {id: series_code, label: series_title}
    hierarchy: [crop, crop_group]
    time: {business: quote_date, period_key: period_ym, range: ['2024-01-01', '2025-12-29'],
           grains: [week, month, quarter, year]}
    measures:
    - {logical: price_mid, band: mid, additive: false, aggregation: avg}
    primary_measure: price_mid
    denominations: [currency, unit]          # never sum or average across these
    scenario: {column: basis, values: {observed: actual, outlook: projected}, default: actual}
synonyms: {price_mid: [mid price, average price], crop: [commodity]}
```

**Use:** put it (or the relevant table) in an LLM's system prompt so it writes correct SQL, or read it in your own code to pick the right column and aggregation:
```python
import yaml
sem = yaml.safe_load(open("artifacts/example/semantic.yaml"))
for name, t in sem["tables"].items():
    if t["kind"] == "measures":
        print(t["view"], "main measure:", t["primary_measure"], "never mix:", t["denominations"])
```

### `structured/views.sql`

**Holds:** one read-only `CREATE VIEW sem_<table>` per source table. The views keep the original columns and add `scenario_label` (`actual` / `projected`).

```sql
CREATE OR REPLACE TEMP VIEW sem_grain_quotes AS
SELECT "series_code" AS "series_code", …, "price_mid" AS "price_mid", …,
       CASE "basis" WHEN 'observed' THEN 'actual' WHEN 'outlook' THEN 'projected' END AS scenario_label
FROM src.main."grain_quotes";
```

**Use:** attach the source database as `src`, run the file, then query the views ([Recipe 1](#1-query-the-views-with-sql)).

### `roles.yaml`

**Holds:** for every column, its role (`entity_id`, `time_business`, `measure_band`, `currency`, `doc_id`, `text_part`, `doc_tag`, `audit`, …), a readable name, a description, a confidence and a source. Each table also gets a kind: `measures`, `text` or `other`.

```yaml
tables:
  farm_bulletins:
    kind: text
    columns:
      bulletin_ref: {role: doc_id, description: Reference of the bulletin., confidence: 0.95, source: inferred}
      crop_tag:     {role: doc_tag, description: Crop the bulletin is tagged with., confidence: 0.95}
      body_part_1:  {role: text_part, part_order: 1, confidence: 0.95}
```

**Use:** generate a data dictionary, or find the article text and date columns in your own pipeline without hard-coding names.

### `entities.yaml`

**Holds:** the catalog, one entry per product, series, group and place found in the measures table: aliases (other names), the table, column and value it came from, parents, places, currencies and units.

```yaml
- id: crop:milling_wheat
  type: Crop
  canonical_name: Milling Wheat
  aliases: [Wheat (milling)]
  attributes:
    sources: [{table: grain_quotes, column: crop, value: Milling Wheat}]
    locations: [location:great_plains]
    parents: [cropgroup:cereals]
  source: metadata
  confidence: 1.0
- id: location:great_plains
  type: Location
  canonical_name: Great Plains
  aliases: [GP]
  attributes: {part_of: location:north_america}
```

**Use:** map user words to the exact SQL filter value (`attributes.sources[].value`), build an autocomplete list, or give an LLM the known names. `aliases` let "GP" and "Great Plains" mean the same place.

### `graph/entities_text.yaml`

**Holds:** entities that appear only in article tags and match nothing in the catalog (for example a crop the price table doesn't cover). Same shape as `entities.yaml`, with `needs_review: true` and `attributes.tag_column`.
**Use:** add them to your entity list for document-only things. Treat them as unconfirmed until reviewed.

### `ontology.yaml`

**Holds:** the kinds of things (`types`, each pointing to the columns it comes from) and how they relate (`relations`: HAS_PARENT, AT_LOCATION, TAGGED_WITH, MENTIONS, AFFECTS…).
**Use:** as a schema for your own graph, or a short "what exists" section in a prompt.

```yaml
types:
- {name: Crop, description: A traded crop., sources: [{table: grain_quotes, columns: [crop]}]}
relations:
- {name: HAS_PARENT, from: Crop, to: CropGroup, table: grain_quotes}
```

### `event_types.yaml`

**Holds:** the kinds of events the news reports, worked out by the LLM from a sample of article summaries. Merges a reviewer made appear as `relabels`.
**Use:** the names as filter values for events, or as categories in your own extraction.

```yaml
event_types:
- {name: supply_disruption, description: Weather or logistics cut the available supply.}
- {name: policy_change, description: A government changes a tariff, quota or rule.}
```

### `structured/moves.jsonl`

**Holds:** one line per unusually large price change: series, date, previous and new values, change in %, z score (standard deviations from the series' own recent history), direction, frequency, currency and unit.

```json
{"id": "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05", "entity_id": "WHT_MILL_GP",
 "date": "2025-05-05", "prev_value": 300.59, "value": 344.25, "change_pct": 14.5, "z": 56.0,
 "direction": "up", "frequency": "weekly", "denomination": {"currency": "USD", "unit": "per_tonne"}}
```

**Use:** alerts, dashboards, or the list of "what needs explaining". `entity_id` is the raw series code; `links/nodes.jsonl` adds the matching catalog id (`catalog_id`).

### `docs/documents.jsonl`

**Holds:** one record per article, built from the text table using the roles: `doc_id`, `date`, `title`, `summary`, `body` (parts joined in order), `tags` (tag column → values) and `meta` (other columns).

```json
{"doc_id": "FB-20250425-01", "date": "2025-04-25", "title": "Drought cuts Great Plains wheat supply",
 "summary": "A prolonged dry spell…", "body": "Drought has cut milling wheat supply…",
 "tags": {"crop_tag": ["Wheat (milling)"], "region_tag": ["GP"]}, "meta": {"outlet": "Farm Ledger"}}
```

**Use:** a clean article corpus for your own retrieval, summarising or extraction, with no need to re-assemble the source rows.

### `docs/chunks.parquet`

**Holds:** each article cut into a head piece (title + summary) and body pieces of about 500 tokens. Columns: `chunk_id`, `doc_id`, `date`, `kind` (head/body), `text`, `title`, `embedding` (a vector from the configured model), `embedding_fingerprint` (which model made it).
**Use:** load into any vector store, or search directly ([Recipe 3](#3-your-own-document-search)). Compare only against query vectors made by the **same** embedding model.

### `docs/fts.duckdb`

**Holds:** a DuckDB database with table `chunks` (the same pieces) and a BM25 full-text index on `text`.
**Use:** keyword search with DuckDB's `fts` extension ([Recipe 3](#3-your-own-document-search)).

### `docs/events.jsonl`

**Holds:** events extracted from the news. Articles about the same event are merged into one, keeping every article and quote. Each event has: type; date (`date_source`: `stated`, or `publication` when the article gave no date); `status` (reported / announced / forecast); summary; entities and places; `doc_ids` quotes; and `affects` (what changes, the aspect such as price, output, demand or cost, the direction, and the exact sentence).

```json
{"event_id": "event:bc1c8d22ee", "type": "supply_disruption", "date": "2025-04-25", "status": "reported",
 "summary": "Drought cuts the Great Plains milling wheat crop.", "entity_ids": ["crop:milling_wheat"],
 "location_ids": ["location:great_plains"], "doc_ids": ["FB-20250425-01", "FB-20250427-01"],
 "quotes": ["Drought has cut milling wheat supply in the Great Plains by about one fifth, …"],
 "affects": [{"entity_id": "crop:milling_wheat", "aspect": "price", "direction": "up", "confidence": 0.8,
              "quote": "Drought has cut milling wheat supply…", "doc_id": "FB-20250425-01"}]}
```

**Use:** an event timeline per product or place, a "what happened" feed, or input to your own causal analysis. Every claim has its sentence, so you can show the proof.

### `graph/nodes.jsonl` and `graph/edges.jsonl`

**Hold:** a property graph.
- **Nodes:** `{id, type, name, props}`. Types are catalog entities, `Document` (`doc:<id>`), `Event` and `Organization`.
- **Edges:** `{src, rel, dst, props, confidence, source, evidence_doc, evidence_quote, needs_review, support, evidence}`. `evidence` lists every supporting article; `support` counts independent articles; `confidence` combines them. An `AFFECTS` edge has `props.aspect` and `props.direction`, and `conflict: true` when articles disagree.

| Relation | From → to | Where it comes from | Evidence |
|---|---|---|---|
| `AFFECTS` | event → thing | article text | sentence(s) |
| `LOCATED_IN` | event → place | article text | sentence |
| `INVOLVES` | event → organisation | article text | sentence |
| `REPORTS` | document → event | article text | sentence |
| `TAGGED_WITH` | document → thing | a tag column value | `column` (no sentence) |
| `MENTIONS` | document → thing | exact name found in the text | none |
| `HAS_PARENT`, `AT_LOCATION`, `PART_OF` | thing → thing | the price table | none |

```json
{"src": "event:bc1c8d22ee", "rel": "AFFECTS", "dst": "crop:milling_wheat",
 "props": {"aspect": "price", "direction": "up"}, "confidence": 0.95, "support": 2, "needs_review": false,
 "evidence": [{"doc": "FB-20250425-01", "quote": "Drought has cut milling wheat supply…", "confidence": 0.8, "direction": "up"},
              {"doc": "FB-20250427-01", "quote": "Importers have started to enquire about cargoes…", "confidence": 0.8, "direction": "up"}]}
```

**Use:** load into NetworkX or Neo4j ([Recipe 2](#2-load-the-graph)), or answer "what affects X?" by filtering edges. The numbers rank evidence strength; they are not probabilities.

### `links/candidate_drivers.jsonl`

**Holds:** for each price change inside the period the news covers, the top 5 events that may explain it. Each line has the score, days before, relation, aspect, direction, the article and the exact quote.

```json
{"move_id": "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05", "event_id": "event:bc1c8d22ee",
 "score": 0.39, "lag_days": 10, "relation": "AFFECTS", "aspect": "price", "direction": "up",
 "evidence_doc": "FB-20250425-01", "evidence_quote": "Drought has cut milling wheat supply…"}
```

**Use:** answer "why did this price move?" with your own agent or report. These are **candidates to read, not proof of cause**. A price change outside the news period never has any.

### `links/nodes.jsonl` and `links/edges.jsonl`

**Hold:** `Move` nodes (with `catalog_id`), `CANDIDATE_DRIVER_OF` edges (event → move), and `HAS_PARENT` edges found from the data (series → product or group).
**Use:** append them to the graph files to get one graph where events connect to price changes.

### `REVIEW.md`

**Holds:** what a person should check, grouped by entity, not by connection: flagged columns, new entities from tags, low-confidence name matches, extraction failures, and decisions that are pending or no longer match. The first lines say which fixes apply instantly (`ctx review apply`) and which need a rerun ([guide](RUNNING.md#fixing-things-later)).

### `review/decisions.yaml`

**Holds:** reviewer decisions, for example "this tag is Great Plains" or "these two events are not the same". The explorer exports it; you can also edit it by hand. Each build applies it, so fixes survive rebuilds.

### `context_explorer.html`

**Holds:** one self-contained offline page (`ctx show`) that loads nothing from the internet. It shows the overview, column meanings, usage rules, entities, price changes with the news behind them, events, a read-only connections explorer and a "needs a human check" list. It contains real data from your build, so share it the way you'd share the files.

**Use:** open it in a browser to look around, or to review items and export decisions ([flow](RUNNING.md#review-flow)).

---

## Recipes

### 1. Query the views with SQL

```python
import duckdb
from pathlib import Path

art = Path("artifacts/example")
con = duckdb.connect()
con.execute("ATTACH 'path/to/source.duckdb' AS src (READ_ONLY)")
con.execute((art / "structured/views.sql").read_text())
print(con.execute("""
    SELECT series_code, date_trunc('month', quote_date) AS month, avg(price_mid) AS price, currency, unit
    FROM sem_grain_quotes
    WHERE scenario_label = 'actual'
    GROUP BY ALL ORDER BY month
""").fetchdf())
```

In plain SQL, from any DuckDB client:
```sql
ATTACH 'path/to/source.duckdb' AS src (READ_ONLY);
.read artifacts/example/structured/views.sql
SELECT * FROM sem_grain_quotes WHERE scenario_label = 'actual' LIMIT 10;
```
Follow `semantic.yaml`: filter `scenario_label`, never mix currencies or units, and use the main measure.

### 2. Load the graph

```python
import json, networkx as nx

def records(path):
    return [json.loads(line) for line in open(path)]

G = nx.MultiDiGraph()
for path in ["artifacts/example/graph/nodes.jsonl", "artifacts/example/links/nodes.jsonl"]:
    for n in records(path):
        G.add_node(n["id"], **n)
for path in ["artifacts/example/graph/edges.jsonl", "artifacts/example/links/edges.jsonl"]:
    for e in records(path):
        if not e["needs_review"]:          # skip unreviewed edges, as our tools do
            G.add_edge(e["src"], e["dst"], **e)

# What affects milling wheat, with proof?
for src, _, e in G.in_edges("crop:milling_wheat", data=True):
    if e["rel"] == "AFFECTS":
        print(G.nodes[src]["name"], e["props"], e["confidence"], e["evidence"][0]["quote"])
```

**Neo4j:** write nodes and edges to two CSVs (`id,type,name` and `src,rel,dst,confidence,quote`) and load them with `LOAD CSV`, or import the JSON lines with APOC (`apoc.load.json`). Use `rel` as the relationship type and `type` as the node label.

### 3. Your own document search

```python
import duckdb, numpy as np, pandas as pd

chunks = pd.read_parquet("artifacts/example/docs/chunks.parquet")
matrix = np.vstack(chunks["embedding"].to_numpy())
query = my_embed("drought wheat supply")              # same model as embedding_fingerprint
best = chunks.iloc[np.argsort(-(matrix @ query))[:5]][["doc_id", "date", "title", "text"]]

con = duckdb.connect("artifacts/example/docs/fts.duckdb", read_only=True)
con.execute("LOAD fts")
keyword = con.execute("""
    SELECT chunk_id, doc_id, title, fts_main_chunks.match_bm25(chunk_id, 'drought wheat') AS score
    FROM chunks WHERE score IS NOT NULL ORDER BY score DESC LIMIT 5
""").fetchdf()
```

### 4. Feed a price change and its drivers to any LLM

```python
import json

moves = {m["id"]: m for m in map(json.loads, open("artifacts/example/structured/moves.jsonl"))}
drivers = [json.loads(l) for l in open("artifacts/example/links/candidate_drivers.jsonl")]
events = {e["event_id"]: e for e in map(json.loads, open("artifacts/example/docs/events.jsonl"))}

move_id = drivers[0]["move_id"]
move = moves[move_id]
lines = [f"Price change: {move['entity_id']} {move['direction']} {move['change_pct']:.1f}% on {move['date']} "
         f"({move['denomination']['currency']} {move['denomination']['unit']})",
         "Candidate news drivers (not proof of cause):"]
for d in sorted((d for d in drivers if d["move_id"] == move_id), key=lambda d: -d["score"]):
    e = events[d["event_id"]]
    lines.append(f"- {e['date']} [{e['status']}] {e['summary']} ({d['aspect']} {d['direction']}, "
                 f"{d['lag_days']} days before) \"{d['evidence_quote']}\" [doc:{d['evidence_doc']}]")
prompt = "\n".join(lines) + "\n\nExplain the change using only these sources and cite [doc:id]."
```

The same pattern works for any agent framework: turn `semantic.yaml` into the system prompt, then give the agent functions that read these files.

---

## Our tools (optional)

Our reference agent uses these tools. They are one way to use the files, a thin layer you can rebuild in your own stack.

| Tool | What it does | Files it reads |
|---|---|---|
| `get_context` | Returns the columns, entities, rules and coverage relevant to a question | `semantic.yaml`, `entities.yaml`, `graph/entities_text.yaml`, `ontology.yaml`, `event_types.yaml`, the graph (neighbours) |
| `resolve_entity` | Turns a name the user wrote into a catalog id (exact, alias, close spelling, then meaning + LLM judge) | `entities.yaml`, `graph/entities_text.yaml`, `entity_embeddings.npy`, `resolve_judge.jsonl` |
| `query_metric` | Runs one of 16 named metrics (latest value, change, trend, volatility…) as safe SQL | `semantic.yaml`, `structured/views.sql`, the source database |
| `validate_sql` / `execute_semantic_sql` | Checks and runs the agent's own SQL, on the `sem_*` views only | `structured/views.sql`, `semantic.yaml` |
| `search_docs` | Search by meaning and keywords, optionally limited to articles about given entities | `docs/chunks.parquet`, `docs/fts.duckdb`, the graph (TAGGED_WITH / MENTIONS) |
| `graph_explore` | Connections around an entity, 1–2 hops, with evidence; flagged edges hidden | `graph/*.jsonl`, `links/edges.jsonl` |
| `explain_move` | Price changes for an entity and period, their candidate drivers (first) and the series values with SQL | `structured/moves.jsonl`, `links/candidate_drivers.jsonl`, `docs/events.jsonl` via the graph, `semantic.yaml` + views |

The tools are plain Python functions in `src/ctxlayer/serving/tools.py` that return JSON-friendly dicts, so you can call them from your own agent.
