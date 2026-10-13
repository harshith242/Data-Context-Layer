"""The frame around REVIEW.md: a header of how to apply fixes and the decisions that do not match."""
import json

from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.pipeline.phases import PHASES
from ctxlayer.pipeline.state import applied_ids

_REVIEW_FILE = "REVIEW.md"
# Phases write the sections between these two markers; the frame is rewritten around them after every build.
_HEADER_END = "<!-- sections -->"
_TAIL_START = "<!-- decisions -->"


def review_header(applied: int, unmatched: int, invalid: int, pending: int = 0) -> str:
    """The first lines of REVIEW.md (spec section 9) with the decision counts."""
    verb = "matches" if unmatched == 1 else "match"
    waiting = f"{pending} pending, " if pending else ""
    return (
        "Fix any time:  ctx review apply  (catalog, rules, moves, events, name matches, merged events, drivers)\n"
        "Needs a rerun from a phase:  column roles -> ctx build --from roles;\n"
        "                             entity types / relations -> --from ontology; "
        "new event types -> --from event-types\n"
        f"Decisions: {applied} applied, {waiting}{unmatched} no longer {verb}, {invalid} invalid "
        "(see the end of this file).\n"
    )


def write_review_frame(pc: PhaseContext, state: dict) -> None:
    """Put the header at the top of REVIEW.md and the unmatched and invalid decisions at the end.

    A decision is unmatched when no phase in `state` applied it, so a phase skipped as up to date keeps its own.
    """
    done = applied_ids(state)
    effective = pc.decisions.effective()
    missed = [decision for ident, decision in effective if ident not in done]
    pending = [decision for decision in missed if _waiting(pc, state, decision)]
    unmatched = [decision for decision in missed if decision not in pending]
    header = review_header(
        len(effective) - len(missed), len(unmatched), len(pc.decisions.invalid), len(pending),
    )

    path = pc.out_dir / _REVIEW_FILE
    body = path.read_text() if path.is_file() else ""
    body = body.split(_HEADER_END, 1)[-1].split(_TAIL_START, 1)[0].strip("\n")
    tail = [
        _TAIL_START, "", "## Pending (run ctx build, or ctx build --from <phase>)", "",
        *_items([_line(d) for d in pending]),
        "", "## Decisions that no longer match", "",
        *_items([_line(d) for d in unmatched]),
        "", "## Invalid decisions", "",
        *_items([f"{item['reason']}: {json.dumps(item['entry'], default=str)}" for item in pc.decisions.invalid]),
    ]
    text = "\n".join([header, _HEADER_END, "", body, "", *tail]) + "\n"
    path.write_text(text)


def _line(decision) -> str:
    return f"{decision.kind.value} {json.dumps(decision.target, sort_keys=True)} ({decision.by}, {decision.at})"


def _waiting(pc: PhaseContext, state: dict, decision) -> bool:
    # Its phase has not run since the decision was written (same test as review._applied).
    phase = next(phase for phase in PHASES if decision.kind in phase.kinds)
    saved = state["phases"].get(phase.name)
    return saved is None or saved["parts"].get("decisions") != pc.decisions.fingerprint(set(phase.kinds))


def _items(lines: list[str]) -> list[str]:
    return [f"- {line}" for line in lines] or ["none"]
