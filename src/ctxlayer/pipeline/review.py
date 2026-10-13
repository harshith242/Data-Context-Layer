"""`ctx review apply`: run the phases that apply the new decisions, list the ones that need a rerun."""
from ctxlayer.common.decisions import Decision, Kind, Verdict
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.pipeline.phases import PHASES
from ctxlayer.pipeline.review_file import review_header  # noqa: F401  (part of this module's interface)
from ctxlayer.pipeline.runner import Review, run_build
from ctxlayer.pipeline.state import load_state

# Kinds whose decisions change later prompts; applying them is a rerun from that phase.
REVIEW_POINT_KINDS = {Kind.COLUMN: "roles", Kind.ENTITY_TYPE: "ontology", Kind.RELATION: "ontology"}
_EVENT_TYPES = "event-types"


def review_apply(pc: PhaseContext) -> dict:
    """Apply the decisions no phase has applied yet, without model calls.

    Runs from the earliest phase that owns a pending instant decision (the phases after it only if stale).
    Returns {"ran": [phase names], "needs_rerun": [{"kind", "target", "command": "ctx build --from <phase>"}]}.
    """
    names = [phase.name for phase in PHASES]
    pending = [decision for decision in _decisions(pc) if not _applied(pc, decision)]
    needs_rerun = []
    instant = []
    for decision in pending:
        point = _review_phase(decision)
        if point is None:
            instant.append(_owner(decision))
            continue
        needs_rerun.append({
            "kind": decision.kind.value, "target": decision.target, "command": f"ctx build --from {point}",
        })

    if not instant:
        return {"ran": [], "needs_rerun": needs_rerun}

    first = min(instant, key=names.index)
    result = run_build(pc, first, None, Review.OFF, force_start=True)
    return {"ran": result["ran"], "needs_rerun": needs_rerun}


def _decisions(pc: PhaseContext) -> list[Decision]:
    return [decision for _, decision in pc.decisions.effective()]


def _owner(decision: Decision) -> str:
    return next(phase.name for phase in PHASES if decision.kind in phase.kinds)


def _review_phase(decision: Decision) -> str | None:
    """The phase to rerun from when the decision changes later prompts, else None (an instant fix)."""
    if decision.kind is Kind.EVENT_TYPE:
        changes_prompt = decision.verdict is Verdict.FIX and "description" in decision.fix
        return _EVENT_TYPES if changes_prompt else None
    return REVIEW_POINT_KINDS.get(decision.kind)


def _applied(pc: PhaseContext, decision: Decision) -> bool:
    """True when the owning phase last ran with the current decisions of its kinds."""
    phase = next(phase for phase in PHASES if decision.kind in phase.kinds)
    saved = load_state(pc.out_dir)["phases"].get(phase.name)
    return saved is not None and saved["parts"].get("decisions") == pc.decisions.fingerprint(set(phase.kinds))
