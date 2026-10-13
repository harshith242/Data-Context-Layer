"""`ctx` command line: LLM check, build, ask, and the evaluation steps."""
import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from ctxlayer.common._names import ANSWERS_DIR, SCORES_DIR
from ctxlayer.common.config import Settings, artifacts_dir, load_settings, require_artifacts
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.embeddings import get_embedder
from ctxlayer.common.llm import LLMError, OllamaLLM
from ctxlayer.common.phase_context import PhaseContext, make_context
from ctxlayer.common.runlog import activate_run, event, finish_run, operation, run_logger
from ctxlayer.evaluation.categorize import categorize
from ctxlayer.evaluation.judge import Judge
from ctxlayer.evaluation.quality import quality_checks
from ctxlayer.evaluation.questions import Question, load_questions
from ctxlayer.evaluation.reference import build_references, resolve_anchor
from ctxlayer.evaluation.report import write_eval_review, write_report
from ctxlayer.evaluation.runner import (
    RUN_LABEL, prepare_windows, run_config, sample_questions, score_config,
)
from ctxlayer.pipeline.phases import PHASES, STRUCTURED_ONLY
from ctxlayer.pipeline.review import review_apply
from ctxlayer.pipeline.runner import Review, adopt, continue_build, phase_status, run_build
from ctxlayer.reference_agent.api import ReferenceAgent

_SUCCESS = 0
_FAILURE = 1
_PING_PROMPT = "Reply with OK."
_DOCUMENTS_ARTIFACT = "docs/documents.jsonl"
_CATEGORIES_FILE = "categories.yaml"
_REFERENCES_FILE = "references.json"
_REVIEW_FILE = "REVIEW.md"
_REPORT_FILE = "report.md"
_PREPARE_HINT = "run `ctx eval prepare` first"


def main(argv: list[str] | None = None) -> int:
    """Run one command; returns the exit code (0 ok, 1 with the message on stderr)."""
    args = _parser().parse_args(argv)
    activate_run()
    try:
        return args.handler(args)
    except (LLMError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return _FAILURE
    finally:
        finish_run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctx", description="Context layer POC.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check-llm", help="one tiny Ollama call").set_defaults(handler=_check_llm)

    show = commands.add_parser("show", help="write an HTML page of everything built for the database")
    show.set_defaults(handler=_show)

    phase_names = [phase.name for phase in PHASES]
    build = commands.add_parser("build", help="run every phase whose inputs changed")
    _add_build_options(build, phase_names)
    build.set_defaults(handler=_build)

    build_all = commands.add_parser("build-all", help="same as build; --after-extract starts at resolve")
    _add_build_options(build_all, phase_names)
    build_all.add_argument(
        "--after-extract", action="store_true",
        help="debug shortcut: reuse the saved extractions and start at resolve (same as --from resolve)",
    )
    build_all.set_defaults(handler=_build)

    commands.add_parser("build-structured", help="same as build --from profile --to moves").set_defaults(
        handler=_build_range, first="profile", last="moves", skip=frozenset(),
    )
    commands.add_parser("build-unstructured", help="same as build --from documents --to graph").set_defaults(
        handler=_build_range, first="documents", last="graph", skip=STRUCTURED_ONLY,
    )
    commands.add_parser("build-links", help="same as build --from links").set_defaults(
        handler=_build_range, first="links", last=None, skip=frozenset(),
    )
    commands.add_parser("status", help="each phase: up to date, stale or never built").set_defaults(handler=_status)
    review = commands.add_parser("review", help="human review decisions")
    review_steps = review.add_subparsers(dest="step", required=True)
    review_steps.add_parser("apply", help="apply review/decisions.yaml").set_defaults(handler=_review_apply)
    commands.add_parser("adopt", help="mark the outputs of an older build as up to date").set_defaults(handler=_adopt)

    ask = commands.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.set_defaults(handler=_ask)

    evaluation = commands.add_parser("eval", help="evaluation steps")
    steps = evaluation.add_subparsers(dest="step", required=True)
    steps.add_parser("prepare", help="categories, references, review").set_defaults(handler=_prepare)
    steps.add_parser("report", help="write eval/report.md").set_defaults(handler=_report)

    run = steps.add_parser("run", help="answer and score the questions")
    run.add_argument("--limit", type=int)
    run.add_argument("--sample", type=int, help="run only N questions picked by category quota (eval/sample.yaml)")
    run.add_argument(
        "--redo", help="comma-separated qids to ask and judge again, e.g. q011,q021; other saved scores are kept",
    )
    run.set_defaults(handler=_run)

    return parser


def _check_llm(args: argparse.Namespace) -> int:
    settings = load_settings()
    logger = run_logger(settings)
    try:
        with operation(logger, "check_llm"):
            reply = OllamaLLM.from_settings(settings).complete(_PING_PROMPT)
    except LLMError as exc:
        model = settings.get("llm.model")
        print(f"FAILED: {exc}\nhint: run `ollama serve`, then `ollama pull {model}`")
        return _FAILURE
    event(logger, "check_llm", latency_s=f"{reply.latency_s:.2f}")
    print(f"ok {reply.latency_s:.2f}s")
    return _SUCCESS


def _show(args: argparse.Namespace) -> int:
    # Imported here so other commands never load the explorer.
    from ctxlayer.serving.explorer import write_explorer

    settings = load_settings()
    require_artifacts(settings)
    path = write_explorer(settings)
    print(f"context explorer -> {path}  (open it in a browser; it contains data from the build)")
    return _SUCCESS


def _add_build_options(parser: argparse.ArgumentParser, phase_names: list[str]) -> None:
    parser.add_argument("--human-review", action="store_true", help="pause after each review point")
    parser.add_argument("--continue", dest="resume", action="store_true", help="resume a paused build")
    parser.add_argument("--from", dest="first", choices=phase_names, help="first phase; it always runs")
    parser.add_argument("--to", dest="last", choices=phase_names, help="last phase")


def _models(settings: Settings) -> PhaseContext:
    require_artifacts(settings)
    llm = OllamaLLM.from_settings(settings)
    return make_context(settings, llm, get_embedder(settings), llm)


def _build(args: argparse.Namespace) -> int:
    settings = load_settings()
    pc = _models(settings)
    logger = run_logger(settings)
    with operation(logger, "build"):
        if args.resume:
            result = continue_build(pc)
        else:
            first = "resolve" if getattr(args, "after_extract", False) else args.first
            review = Review.STOP if args.human_review else Review.OFF
            result = run_build(pc, first, args.last, review, force_start=first is not None)
    return _report_build(logger, result)


def _build_range(args: argparse.Namespace) -> int:
    settings = load_settings()
    pc = _models(settings)
    logger = run_logger(settings)
    with operation(logger, "build"):
        result = run_build(pc, args.first, args.last, Review.OFF, force_start=True, skip=args.skip)
    return _report_build(logger, result)


def _report_build(logger, result: dict) -> int:
    event(logger, "build", ran=len(result["ran"]), skipped=len(result["skipped"]))
    print(f"ran: {', '.join(result['ran']) or 'none'}")
    print(f"skipped (up to date): {', '.join(result['skipped']) or 'none'}")
    print(f"extraction cache: hits={result['llm']['hits']} misses={result['llm']['misses']}")
    if result["paused_after"]:
        print(f"paused after {result['paused_after']}: review in ctx show, record decisions, then ctx build --continue")
    return _SUCCESS


def _status(args: argparse.Namespace) -> int:
    settings = load_settings()
    require_artifacts(settings)
    for row in phase_status(make_context(settings)):
        print(f"{row['phase']:<12} {row['state']:<12} {row['reason']}".rstrip())
    return _SUCCESS


def _review_apply(args: argparse.Namespace) -> int:
    settings = load_settings()
    pc = _models(settings)
    with operation(run_logger(settings), "review_apply"):
        result = review_apply(pc)
    print(f"ran: {', '.join(result['ran']) or 'none'}")
    for item in result["needs_rerun"]:
        print(f"needs a rerun: {item['kind']} {json.dumps(item['target'], sort_keys=True)}: {item['command']}")
    return _SUCCESS


def _adopt(args: argparse.Namespace) -> int:
    settings = load_settings()
    require_artifacts(settings)
    result = adopt(make_context(settings))
    print(f"adopted: {', '.join(result['adopted']) or 'none'}")
    for name, files in result["missing"].items():
        print(f"not adopted, missing outputs: {name}: {', '.join(files)}")
    return _SUCCESS


def _ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    require_artifacts(settings)
    logger = run_logger(settings)
    with operation(logger, "ask"):
        with contextlib.closing(ReferenceAgent(settings)) as agent:
            answer = agent.answer(args.question)
    if answer.error:
        event(logger, "ask", result="error")
        print(f"error: {answer.error}", file=sys.stderr)
        return _FAILURE

    event(logger, "ask", tool_calls=len(getattr(answer, "tool_trace", [])))
    print(answer.text)
    for citation in answer.citations:
        print(f"[{citation['kind']}] {_citation_text(citation)}")
    return _SUCCESS


def _citation_text(citation: dict) -> str:
    # {"kind": "row", "table": "t", "rowid": 3} -> "t#3"; sql citations show the query.
    if citation["kind"] == "row":
        return f"{citation['table']}#{citation['rowid']}"
    if citation["kind"] == "doc":
        return f"doc:{citation['doc_id']}"
    return citation["sql"]


def _prepare(args: argparse.Namespace) -> int:
    settings = load_settings()
    eval_dir = settings.path("paths.eval")
    questions = _load_questions(settings)
    logger = run_logger(settings)

    # Categorising needs the LLM; the reference queries run on the local database.
    llm = OllamaLLM.from_settings(settings)
    with operation(logger, "eval_prepare", questions=len(questions)):
        categories = categorize(questions, llm, eval_dir / _CATEGORIES_FILE, workers=settings.get("llm.parallel_calls"))
        with contextlib.closing(connect_readonly(settings.path("database.path"))) as con:
            references = build_references(questions, con, resolve_anchor(settings), settings)

        (eval_dir / _REFERENCES_FILE).write_text(json.dumps(references, indent=1, default=str))
        write_eval_review(categories, references, eval_dir / _REVIEW_FILE)

    by_category = Counter(entry["category"] for entry in categories.values())
    print(f"questions: {len(questions)}")
    print("categories: " + ", ".join(f"{name}={count}" for name, count in sorted(by_category.items())))
    print(f"needs review: {sum(bool(entry.get('needs_review')) for entry in categories.values())}")
    print(_reference_counts(references))
    print(_window_counts(settings, questions, llm))
    print(f"wrote {eval_dir / _CATEGORIES_FILE}, {eval_dir / _REFERENCES_FILE}, {eval_dir / _REVIEW_FILE}")
    event(logger, "eval_prepare", questions=len(questions), needs_review=sum(
        bool(entry.get("needs_review")) for entry in categories.values()
    ))
    return _SUCCESS


def _window_counts(settings: Settings, questions: list[Question], llm: OllamaLLM) -> str:
    # Question periods are judged against the document window, so they need the unstructured build.
    if not (artifacts_dir(settings) / _DOCUMENTS_ARTIFACT).is_file():
        return f"windows: skipped, {_DOCUMENTS_ARTIFACT} not found; run `ctx build-all` first"

    counts = prepare_windows(settings, questions, llm)
    return "windows: " + ", ".join(f"{name}={count}" for name, count in counts.items())


def _reference_counts(references: dict) -> str:
    results = [(name, result) for ref in references.values() for name, result in ref["baselines"].items() if result]
    errors = Counter(name for name, result in results if result["error"] is not None)
    error_counts = ", ".join(f"{name}={errors[name]}" for name in dict.fromkeys(name for name, _ in results))
    return (
        f"references: {len(results)} queries, errors {error_counts or 'none'}, "
        f"transpiled={sum(result['transpiled'] for _, result in results)}, "
        f"re-anchored={sum(result['reanchored'] for _, result in results)}, "
        f"current-date-not-anchored={sum(result['uses_wall_clock'] for _, result in results)}, "
        f"disagreements={sum(ref['agree'] is False for ref in references.values())}"
    )


def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    require_artifacts(settings)
    eval_dir = settings.path("paths.eval")
    questions = _load_questions(settings)
    categories = _load_categories(eval_dir)
    references = _load_references(eval_dir)
    stale = [q.qid for q in questions if references.get(q.qid, {}).get("question") not in (None, q.text)]
    if stale:
        raise ValueError(f"references.json does not match the questions for {', '.join(stale[:5])}; {_PREPARE_HINT}")
    logger = run_logger(settings)

    if args.sample:
        questions = sample_questions(questions, categories, eval_dir, args.sample)
        print(f"sample: {len(questions)} questions -> {eval_dir / 'sample.yaml'}")

    label = RUN_LABEL
    redo = {qid.strip() for qid in args.redo.split(",") if qid.strip()} if args.redo else None
    requested = len(questions[:args.limit])
    with operation(logger, "eval_run", label=label, questions=requested):
        answers = run_config(
            questions, lambda: ReferenceAgent(settings), label, eval_dir, args.limit,
            settings.get("llm.parallel_calls"), redo,
        )

        judge_llm = OllamaLLM.from_settings(settings)
        scores = score_config(
            questions, categories, references, answers, Judge(judge_llm), judge_llm, settings, label, redo,
        )

    statuses = Counter(score["exec"]["status"] for score in scores.values())
    print(
        f"{label}: answered {len(answers)}, errors {sum(score['error'] for score in scores.values())}, "
        f"exec pass={statuses['pass']} fail={statuses['fail']} skipped={statuses['skipped']}"
    )
    print(f"wrote {eval_dir / SCORES_DIR / (label + '.json')}")
    event(logger, "eval_run", label=label, answered=len(answers), errors=sum(score["error"] for score in scores.values()))
    return _SUCCESS


def _report(args: argparse.Namespace) -> int:
    settings = load_settings()
    eval_dir = settings.path("paths.eval")
    questions = _load_questions(settings)
    categories = _load_categories(eval_dir)
    logger = run_logger(settings)

    # Every scores file is one column.
    scores_by_label = {path.stem: _read_json(path) for path in sorted((eval_dir / SCORES_DIR).glob("*.json"))}
    answers_by_label = {
        label: {path.stem: _read_json(path) for path in (eval_dir / ANSWERS_DIR / label).glob("*.json")}
        for label in scores_by_label
    }

    with operation(logger, "eval_report", runs=len(scores_by_label)):
        out_path = eval_dir / _REPORT_FILE
        quality = quality_checks(settings)
        write_report(
            questions, categories, scores_by_label, answers_by_label, out_path, quality=quality,
            database=artifacts_dir(settings).name,
        )

    event(logger, "eval_report", runs=len(scores_by_label), quality_checks=len(quality))
    print(f"report: {len(scores_by_label)} runs, {len(questions)} questions -> {out_path}")
    return _SUCCESS


def _load_questions(settings: Settings) -> list[Question]:
    return load_questions(
        settings.path("eval.questions_csv"), settings.get("eval.question_column"), settings.get("eval.baselines")
    )


def _load_categories(eval_dir: Path) -> dict:
    path = eval_dir / _CATEGORIES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; {_PREPARE_HINT}")
    return yaml.safe_load(path.read_text())


def _load_references(eval_dir: Path) -> dict:
    path = eval_dir / _REFERENCES_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; {_PREPARE_HINT}")
    return _read_json(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())
