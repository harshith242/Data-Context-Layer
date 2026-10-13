"""Markdown reports: eval/report.md (ids and aggregates only) and eval/REVIEW.md."""
import statistics
from pathlib import Path
from typing import Callable

from ctxlayer.common._names import ANSWERS_DIR, NO_MAJORITY, OURS
from ctxlayer.common.files import read_yaml
from ctxlayer.evaluation.categorize import CATEGORIES
from ctxlayer.evaluation.exec_check import ExecStatus
from ctxlayer.evaluation.questions import Question

_NONE = "-"
_LOW_CONFIDENCE = 0.7
_HUMAN = "human"
_FAILED_VERDICTS = ("incorrect", "unusable")
_MS_PER_S = 1000
_WINDOWS_FILE = "windows.yaml"
def write_report(
    questions: list[Question],
    categories: dict,
    scores_by_label: dict[str, dict],
    answers_by_label: dict[str, dict],
    out_path: Path,
    quality: dict,
    database: str,
) -> None:
    """Per category tables, a pairwise table per baseline, window / citation / quality sections
    and the failure list, as markdown. `database` names the artifacts folder the review link points to."""
    labels = sorted(scores_by_label)
    groups = _qids_by_category(questions, categories)
    counts = [[category, str(len(qids))] for category, qids in groups.items()]

    def rates(title: str, hit: Callable[[dict], bool | None]) -> str:
        return _rate_table(title, hit, labels, groups, scores_by_label)

    def medians(title: str, measure: Callable[[dict], float]) -> str:
        rows = [
            [category, *_run_cells(measure, labels, qids, scores_by_label, answers_by_label)]
            for category, qids in groups.items()
        ]
        return _titled(title, ["category", *labels], rows)

    names = _baseline_names(questions)
    sections = [
        "# Evaluation report",
        "## Results per category",
        _titled("Questions", ["category", "n"], counts),
        rates("Execution pass", _exec_hit),
        rates("Correct", _correct_hit),
        rates("Behaviour pass", _behaviour_hit),
        _latency_table(questions, names, labels, groups, scores_by_label, answers_by_label),
        medians("Median tool calls", _tool_calls),
        medians("Median est. tokens", _est_tokens),
        *(
            section
            for name in names
            for section in (
                f"## Pairwise vs {name}",
                _pairwise_table(name, groups, scores_by_label),
                _position_check(name, scores_by_label),
            )
        ),
        *_final_sections(names, labels, scores_by_label, out_path.parent, quality, database),
        "## Failures",
        _failures_table(scores_by_label),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections) + "\n")


def _final_sections(names: list[str], labels: list[str], scores_by_label: dict, eval_dir: Path, quality: dict, database: str) -> list[str]:
    # In-window subset, citations, quality, REVIEW link; absent inputs skip a section.
    sections = []

    windows_path = eval_dir / _WINDOWS_FILE
    if windows_path.is_file():
        windows = read_yaml(windows_path) or {}
        inside = {qid for qid, window in windows.items() if window["in_doc_window"]}
        sections += [f"## Questions inside the document window (n={len(inside)})", _metric_table(names, labels, scores_by_label, inside)]

    cited = _citation_rows(labels, scores_by_label)
    if cited:
        sections += ["## Citation validity", _table(["run", "valid/total"], cited)]

    checklist = _checklist_rows(labels, scores_by_label)
    if checklist:
        sections += ["## Driver checklist", _table(["run", "covered/total"], checklist)]

    if quality:
        rows = [[name, str(count)] for name, count in sorted(quality.items())]
        sections += ["## Context-layer quality", _table(["check", "count"], rows)]

    sections += ["## Context-layer review files", f"- [{database}](../artifacts/{database}/REVIEW.md)"]

    return sections


def _metric_table(names: list[str], labels: list[str], scores_by_label: dict, qids: set[str] | None) -> str:
    # One row per label; `qids` None = every scored question. One pairwise column per baseline.
    hits = [_exec_hit, _correct_hit, _behaviour_hit, *(_pairwise_win(name) for name in names)]
    rows = []
    for label in labels:
        scores = [score for qid, score in scores_by_label[label].items() if qids is None or qid in qids]
        rows.append([label, *(_percent(hit, scores) for hit in hits)])
    header = ["run", "exec pass %", "correct %", "behaviour pass %", *(f"pairwise win % vs {name}" for name in names)]
    return _table(header, rows)


def _percent(hit: Callable[[dict], bool | None], scores: list[dict]) -> str:
    # "66.7% (2/3)"; questions where `hit` is None are left out.
    counted = [result for result in (hit(score) for score in scores) if result is not None]
    if not counted:
        return _NONE
    return f"{100 * sum(counted) / len(counted):.1f}% ({sum(counted)}/{len(counted)})"


def _pairwise_win(name: str) -> Callable[[dict], bool | None]:
    def win(score: dict) -> bool | None:
        result = score["pairwise"].get(name)
        if not result:
            return None
        return result["verdict"] == OURS

    return win


def _citation_rows(labels: list[str], scores_by_label: dict) -> list[list[str]]:
    rows = []
    for label in labels:
        checks = [score["citations"] for score in scores_by_label[label].values() if "citations" in score]
        if checks:
            rows.append([label, f"{sum(c['valid'] for c in checks)}/{sum(c['total'] for c in checks)}"])
    return rows


def _checklist_rows(labels: list[str], scores_by_label: dict) -> list[list[str]]:
    rows = []
    for label in labels:
        checks = [score["checklist"] for score in scores_by_label[label].values() if "checklist" in score]
        if checks:
            rows.append([label, f"{sum(c['covered'] for c in checks)}/{sum(c['total'] for c in checks)}"])
    return rows


def write_eval_review(categories: dict, references: dict, out_path: Path) -> None:
    """Items for a human to check: weak categories, reference problems, adjusted queries."""
    low = [
        f"- {qid} ({entry['category']}, confidence {entry.get('confidence')}): {entry['question']}"
        for qid, entry in categories.items()
        if _needs_category_review(entry)
    ]
    disagree = [f"- {qid}" for qid, ref in references.items() if ref["agree"] is False]
    reanchored = _contestant_items(references, lambda result: result["reanchored"])
    unanchored = _contestant_items(references, lambda result: result["uses_wall_clock"])
    transpiled = _contestant_items(references, lambda result: result["transpiled"])
    # The error text may quote data; it stays in references.json.
    errors = _contestant_items(references, lambda result: result["error"] is not None)

    sections = [
        "# Evaluation review",
        _bullet_section("Low-confidence categories", low),
        _bullet_section("Baselines disagree", disagree),
        _bullet_section("Re-anchored to the data date", reanchored),
        _bullet_section("Uses current date, not re-anchored", unanchored),
        _bullet_section("Transpiled for DuckDB", transpiled),
        _bullet_section("Reference errors (details in references.json)", errors),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections) + "\n")


def _baseline_names(questions: list[Question]) -> list[str]:
    # Baselines are the same for every question.
    return list(questions[0].baselines) if questions else []


def _qids_by_category(questions: list[Question], categories: dict) -> dict[str, list[str]]:
    # {"lookup": ["q001", "q002"], ...} in CATEGORIES order, present categories only.
    grouped: dict[str, list[str]] = {}
    for question in questions:
        grouped.setdefault(categories[question.qid]["category"], []).append(question.qid)
    return {category: grouped[category] for category in CATEGORIES if category in grouped}


def _titled(title: str, header: list[str], rows: list[list[str]]) -> str:
    return f"### {title}\n\n{_table(header, rows)}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def _rate(hits: int, total: int) -> str:
    # "67% (2/3)"
    if total == 0:
        return _NONE
    return f"{100 * hits / total:.0f}% ({hits}/{total})"


def _median(values: list[float]) -> str:
    if not values:
        return _NONE
    return f"{statistics.median(values):.1f}"


def _exec_hit(score: dict) -> bool | None:
    status = score["exec"]["status"]
    if status == ExecStatus.SKIPPED.value:
        return None
    return status == ExecStatus.PASS.value


def _correct_hit(score: dict) -> bool | None:
    if score["correctness"] is None:
        return None
    return score["correctness"]["verdict"] == "correct"


def _behaviour_hit(score: dict) -> bool | None:
    if score["behaviour"] is None:
        return None
    return score["behaviour"]["pass"]


def _latency(answer: dict) -> float:
    return answer["latency_s"]


def _tool_calls(answer: dict) -> float:
    return len(answer["tool_trace"])


def _est_tokens(answer: dict) -> float:
    return answer["tokens"]["est_tokens"]


def _rate_table(title: str, hit: Callable[[dict], bool | None], labels, groups, scores_by_label) -> str:
    # `hit(score)` is True / False, or None to leave the question out of the rate.
    rows = []
    for category, qids in groups.items():
        cells = []
        for label in labels:
            scores = scores_by_label[label]
            results = [hit(scores[qid]) for qid in qids if qid in scores]
            counted = [result for result in results if result is not None]
            cells.append(_rate(sum(counted), len(counted)))
        rows.append([category, *cells])
    return _titled(title, ["category", *labels], rows)


def _run_cells(measure, labels, qids, scores_by_label, answers_by_label) -> list[str]:
    # One median per label over the category's scored questions.
    cells = []
    for label in labels:
        answers = answers_by_label[label]
        values = [measure(answers[qid]) for qid in qids if qid in scores_by_label[label]]
        cells.append(_median(values))
    return cells


def _latency_table(questions, names, labels, groups, scores_by_label, answers_by_label) -> str:
    # Baseline times come from the questions file (ms -> s), the rest from our runs.
    by_qid = {question.qid: question for question in questions}
    rows = []
    for category, qids in groups.items():
        theirs = [
            _median([
                by_qid[qid].baselines[name].time_ms / _MS_PER_S
                for qid in qids if by_qid[qid].baselines[name].time_ms is not None
            ])
            for name in names
        ]
        ours = _run_cells(_latency, labels, qids, scores_by_label, answers_by_label)
        rows.append([category, *theirs, *ours])
    return _titled("Median latency (s)", ["category", *names, *labels], rows)


def _pairwise_table(opponent: str, qids_by_category: dict, scores_by_label: dict) -> str:
    header = [
        "category", "run", "n", OURS, opponent, NO_MAJORITY, "ours win rate",
        "traps ours", f"traps {opponent}",
    ]
    rows = []

    for category, qids in qids_by_category.items():
        for label, scores in sorted(scores_by_label.items()):
            results = [scores[qid]["pairwise"][opponent] for qid in qids if qid in scores and scores[qid]["pairwise"].get(opponent)]
            if not results:
                continue

            verdicts = [result["verdict"] for result in results]
            wins = verdicts.count(OURS)
            # Errored answers lose without a judge call, so they carry no trap counts.
            traps = [result.get("traps", {}) for result in results]
            rows.append([
                category, label, str(len(verdicts)), str(wins), str(verdicts.count(opponent)),
                str(verdicts.count(NO_MAJORITY)), _rate(wins, len(verdicts)),
                str(sum(trap.get(OURS, 0) for trap in traps)),
                str(sum(trap.get("opponent", 0) for trap in traps)),
            ])

    return _table(header, rows)


def _position_check(name: str, scores_by_label: dict) -> str:
    """Per config, from each question's first pass: how often ours went first, and our win rate by position.

    A large gap between the two rates means the judge favours a position, e.g. 61% first vs 40% second."""
    lines = []
    for label, scores in sorted(scores_by_label.items()):
        firsts = [result["passes"][0] for result in (score["pairwise"].get(name) for score in scores.values()) if result and result.get("passes")]
        if not firsts:
            continue
        shown_first = [item for item in firsts if item["ours_first"]]
        shown_second = [item for item in firsts if not item["ours_first"]]
        lines.append(
            f"- {label}: ours shown first {len(shown_first)}/{len(firsts)}; "
            f"win rate when first {_rate(_wins(shown_first), len(shown_first))}, "
            f"when second {_rate(_wins(shown_second), len(shown_second))}"
        )
    return "Position check:\n" + "\n".join(lines) if lines else ""


def _wins(passes: list[dict]) -> int:
    return sum(item["winner"] == OURS for item in passes)


def _failures_table(scores_by_label: dict) -> str:
    rows = []
    for label, scores in sorted(scores_by_label.items()):
        for qid, score in sorted(scores.items()):
            reasons = _failure_reasons(score)
            if reasons:
                rows.append([qid, score["category"], label, "; ".join(reasons), f"{ANSWERS_DIR}/{label}/{qid}.json"])
    return _table(["qid", "category", "run", "reason", "answer and tool trace"], rows)


def _failure_reasons(score: dict) -> list[str]:
    # Codes only: judge sentences may quote the answer.
    if score["error"]:
        return ["error"]

    reasons = []
    if score["exec"]["status"] == ExecStatus.FAIL.value:
        reasons.append(f"exec: {score['exec']['reason']}")
    if score["correctness"] and score["correctness"]["verdict"] in _FAILED_VERDICTS:
        reasons.append(f"correctness: {score['correctness']['verdict']}")
    if score["behaviour"] and not score["behaviour"]["pass"]:
        reasons.append("behaviour: fail")
    return reasons


def _needs_category_review(entry: dict) -> bool:
    if entry.get("source") == _HUMAN:
        return False
    return bool(entry.get("needs_review")) or entry.get("confidence", 0) < _LOW_CONFIDENCE


def _contestant_items(references: dict, picked: Callable[[dict], bool]) -> list[str]:
    # "- q001 (a, b)": the baselines whose reference result matches `picked`.
    items = []
    for qid, ref in references.items():
        names = [name for name, result in ref["baselines"].items() if result is not None and picked(result)]
        if names:
            items.append(f"- {qid} ({', '.join(names)})")
    return items


def _bullet_section(title: str, items: list[str]) -> str:
    return "\n".join([f"## {title}", "", *(items or ["- none"])])
