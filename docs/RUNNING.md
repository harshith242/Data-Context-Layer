# Running the context layer

Setup, commands, review flow and troubleshooting. Overview: [README](../README.md). Output files: [CONTEXT_FILES.md](CONTEXT_FILES.md).

---

## Setup

Needs Python 3.12 and [Ollama](https://ollama.com).

```bash
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"
ollama pull qwen3.8:27b
ollama pull qwen3-embedding:0.6b
ollama serve                        # keep it running
ctx check-llm                       # one tiny LLM call, to confirm Ollama and the model work
```

### Settings you may change

All settings are in `config/settings.yaml`.

| Setting | Default | Meaning |
|---|---|---|
| `llm.model` | `qwen3.8:27b` | The Ollama model used by every LLM step |
| `llm.base_url` | `http://localhost:11434` | Where Ollama listens |
| `llm.parallel_calls`, `llm.timeout_s` | `4`, `300` | Parallel LLM calls and per-call timeout in seconds |
| `embeddings.model` | `qwen3-embedding:0.6b` | The embedding model (run by the same Ollama) |
| `database.path` | `data/example.duckdb` | Your DuckDB file. Output goes to `artifacts/<database file stem>/` |
| `eval.questions_csv`, `eval.question_column` | `examples/questions.csv`, `question` | Your questions for evaluation |
| `eval.baselines` | one example entry | Your reference systems (see below) |
| `review.confidence_threshold` | `0.8` | Items below this are flagged `needs_review` |

### Quick start with the example

```bash
python examples/make_example_db.py      # writes data/example.duckdb and examples/questions.csv
mkdir -p sources/example && cp examples/context.md sources/example/context.md   # optional prior knowledge
ctx build                               # or: ctx build --human-review
ctx show                                # context explorer page
ctx ask "Why did the milling wheat price jump in May 2025?"
```

### Using your own data

1. Set `database.path` to your DuckDB file.
2. Optional: add `sources/<database file stem>/context.md` (and `<table>.context.md` for one table). Use `examples/context.md` as a template. If the file exists, every LLM step reads it. Optional headings let each step read only its part: `## Columns` (what columns mean), `## Names` (other names for the same thing), `## Currencies` (units and conversions), `## News` (what the documents are about). Text before the first heading goes to every step.
3. Optional, for evaluation: make a questions CSV, set `eval.questions_csv` and `eval.question_column`, and list your reference systems in `eval.baselines`:

```yaml
eval:
  questions_csv: my/questions.csv
  question_column: question
  baselines:
    - {name: my_system, sql_column: my_system_sql, answer_column: my_system_answer, time_ms_column: my_system_time_ms}
```

Only `name` is required, and zero baselines is fine. The first baseline with a `sql_column` is the reference for number and correctness scores. Each baseline with an `answer_column` is compared with the agent's answer. `dialect` (for example `spark`) retries SQL that fails on DuckDB after translating it from that dialect.

---

## Commands

Every command prints counts and paths only (except `ask`). Output goes to `artifacts/<database file stem>/`.

| Command | What it does |
|---|---|
| `ctx check-llm` | One tiny Ollama call |
| `ctx build` | Runs every phase whose inputs changed (all of them on a first build), then writes the explorer |
| `ctx build --human-review` | Same, but pauses after `roles` and after `event-types` |
| `ctx build --continue` | Resumes a paused build |
| `ctx build --from <phase> [--to <phase>]` | Runs that range. `--from` always runs; later phases run only if stale |
| `ctx build-structured` | Shortcut for `ctx build --from profile --to moves` |
| `ctx build-unstructured` | Shortcut for `ctx build --from documents --to graph` |
| `ctx build-links` | Shortcut for `ctx build --from links` |
| `ctx build-all` | Alias of `ctx build`. `--after-extract` is the same as `--from resolve` |
| `ctx status` | Each phase: up to date, stale or never built, with the reason |
| `ctx review apply` | Applies `review/decisions.yaml` and reruns what it affects |
| `ctx adopt` | Marks existing outputs as up to date, without running anything |
| `ctx show` | Writes the context explorer page |
| `ctx ask "question"` | One answer with citations |
| `ctx eval prepare` | Question categories and baseline SQL results |
| `ctx eval run [--sample K] [--limit K] [--redo q1,q2]` | Answers and scores; resumes finished answers. `--redo` redoes only those questions |
| `ctx eval report` | Writes `eval/report.md` |

Phases, in order: `profile, roles, ontology, documents, index, event-types, catalog, semantic, views, moves, extraction, resolve, dedupe, graph, links`. The review points are `roles` and `event-types`.

### Typical evaluation run

```bash
ctx build
ctx eval prepare
ctx eval run --sample 15
ctx eval report
```

Answers go to `eval/answers/agent/`, scores to `eval/scores/agent.json`. `--sample 15` picks questions across categories and saves the pick to `eval/sample.yaml`; edit that file to change the set.

---

## Review flow

1. `ctx build --human-review` runs, then **pauses after roles** and writes the explorer.
2. Open the explorer (`ctx show`). Enter your name, then mark each item correct, wrong or fix. Add a note if useful.
3. Click **Export decisions** and save the file as `review/decisions.yaml`. The export merges the decisions already on the page with yours; the newest wins. The pipeline never writes this file.
4. `ctx build --continue` applies the decisions and runs on. It **pauses again after event-types**. Repeat steps 2 and 3.
5. `ctx build --continue` finishes the build.

`ctx status` shows what is stale and why. `REVIEW.md` lists what to check after each build.

### Fixing things later

Edit or export `review/decisions.yaml`, then run `ctx review apply`. It reruns only what your decisions affect, and lists any that need a `--from` rerun, with the command.

| Decision kind | Effect | Rerun |
|---|---|---|
| `entity`, `rule`, `move`, `event`, `name_match`, `new_entity`, `same_event`, `driver` | Applied at once | `ctx review apply` |
| `column` role change | Later prompts change | `ctx build --from roles` |
| `entity_type`, `relation` | Later prompts change | `ctx build --from ontology` |
| a new `event_type` | Later prompts change | `ctx build --from event-types` |
| renaming or merging an `event_type` | Only relabels events | none |

---

## Time

Everything runs on your machine with a local model, so there is no per-call fee. Speed depends on your hardware and model. The slowest steps are extraction (one call per document) and the agent and judges in `eval run`. Embeddings are quick.

Time goes up with more documents, a larger `llm.model`, or more questions in `eval run`. Changing `llm.model`, roles, ontology, event types or `context.md` Names/News re-extracts every document. Unchanged rebuilds reuse cached LLM answers. The `runs/` folder is only a log and is safe to delete.

---

## Troubleshooting

- **Ollama not running.** `ctx check-llm` fails or times out. Start it with `ollama serve` (keep that terminal open) and check `llm.base_url` matches where it listens.
- **Model missing.** Errors mention a model not found. Run `ollama pull qwen3.8:27b` and `ollama pull qwen3-embedding:0.6b`, or set `llm.model` / `embeddings.model` to a model you have (`ollama list`).
- **Calls time out on a big model.** Raise `llm.timeout_s` or lower `llm.parallel_calls`.
- **A phase did not rerun.** Run `ctx status` to see why; use `ctx build --from <phase>` to force it.
- **Moves show 0 drivers.** Drivers exist only inside the period the documents cover.
