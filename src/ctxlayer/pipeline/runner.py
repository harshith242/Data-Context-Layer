"""Run phases in order, skipping the up to date ones; report their status; adopt an older build."""
from enum import Enum

from ctxlayer.common.decisions import Decisions
from ctxlayer.common.files import read_json, read_yaml
from ctxlayer.common.ontology import restamp_legacy_version
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.pipeline.phases import PHASES, Phase
from ctxlayer.pipeline.review_file import write_review_frame
from ctxlayer.pipeline.state import (
    HUMAN_REVIEW, PAUSED_AFTER, STATE_FILE, changed_reason, load_state, parts, record, save_state,
)
from ctxlayer.unstructured.phases import run_event_types

_UP_TO_DATE = "up to date"
_STALE = "stale"
_NEVER_BUILT = "never built"
_UPSTREAM_STALE = "upstream stale"
_NO_PAUSE = "no paused build; run ctx build"


class Review(Enum):
    OFF = "off"
    STOP = "stop"  # pause after each review-point phase that ran


def phase_status(pc: PhaseContext) -> list[dict]:
    """[{phase, state: "up to date"|"stale"|"never built", reason}] in build order.

    A phase is stale when its own inputs changed since it last ran, or a phase it reads from is not up to date.
    """
    saved = load_state(pc.out_dir)["phases"]
    states: dict[str, str] = {}
    rows = []
    for phase in PHASES:
        state, reason = _status(phase, pc, saved.get(phase.name), states)
        states[phase.name] = state
        rows.append({"phase": phase.name, "state": state, "reason": reason})
    return rows


def run_build(
    pc: PhaseContext, start: str | None, stop: str | None, review: Review, force_start: bool,
    skip: frozenset[str] = frozenset(),
) -> dict:
    """Run the stale phases from `start` to `stop` (default: all), recording build_state.json after each.

    `force_start` runs the first phase even when up to date, and so every phase whose inputs changed.
    Phases in `skip` are left out. With Review.STOP the build pauses after a review-point phase that ran.
    Returns {"ran", "skipped", "paused_after", "llm": {"hits", "misses"}} (extraction cache use).
    """
    unrecorded = not (pc.out_dir / STATE_FILE).is_file() and (pc.out_dir / "ontology.yaml").is_file()
    if unrecorded and start != "profile":
        raise ValueError(
            "this folder was built before build_state.json: run ctx adopt first (or ctx build --from profile to rebuild)"
        )

    state = load_state(pc.out_dir)
    state.pop(PAUSED_AFTER, None)
    state.pop(HUMAN_REVIEW, None)
    result = _execute(pc, state, start, stop, review, force_start, skip)
    _finish(pc, state, result["paused_after"])
    return result


def continue_build(pc: PhaseContext) -> dict:
    """Resume after the phase a --human-review build paused at; later review points pause again.

    Phases at or before the pause that went stale meanwhile (e.g. a roles decision made during the pause) run first.
    """
    state = load_state(pc.out_dir)
    paused = state.get(PAUSED_AFTER)
    if not paused:
        raise ValueError(_NO_PAUSE)

    names = [phase.name for phase in PHASES]
    review = Review.STOP if state.get(HUMAN_REVIEW) else Review.OFF
    state.pop(PAUSED_AFTER, None)
    state.pop(HUMAN_REVIEW, None)
    earlier = _execute(pc, state, None, paused, Review.OFF, False, frozenset())
    later = _execute(pc, state, names[names.index(paused) + 1], None, review, False, frozenset())

    result = {
        "ran": earlier["ran"] + later["ran"], "skipped": earlier["skipped"] + later["skipped"],
        "paused_after": later["paused_after"],
        "llm": {key: earlier["llm"][key] + later["llm"][key] for key in earlier["llm"]},
    }
    _finish(pc, state, result["paused_after"])
    return result


def adopt(pc: PhaseContext) -> dict:
    """Record the phases whose outputs all exist as up to date, running none of them.

    For builds made before build_state.json: they are not rerun, so roles and the ontology are not asked again.
    Returns {"adopted": [phase names], "missing": {phase name: [missing output files]}}.
    """
    _move_old_event_types(pc)
    _restamp_ontology(pc)
    state = load_state(pc.out_dir)
    adopted: list[str] = []
    missing: dict[str, list[str]] = {}
    for phase in PHASES:
        absent = [name for name in phase.outputs if not (pc.out_dir / name).is_file()]
        if absent:
            missing[phase.name] = absent
            continue
        # Decisions already in review/decisions.yaml were not applied by this build.
        found = {**parts(phase, pc), "decisions": Decisions.empty().fingerprint(set(phase.kinds))}
        record(state, phase, found)
        adopted.append(phase.name)

    save_state(pc.out_dir, state)
    return {"adopted": adopted, "missing": missing}


def _execute(
    pc: PhaseContext, state: dict, start: str | None, stop: str | None, review: Review, force_start: bool,
    skip: frozenset[str],
) -> dict:
    # The phase loop of run_build; `state` is updated and saved after each phase.
    first, last = _range(start, stop)
    ran: list[str] = []
    skipped: list[str] = []
    cache = {"hits": 0, "misses": 0}
    paused = None
    for index in range(first, last + 1):
        phase = PHASES[index]
        found = parts(phase, pc)
        forced = force_start and index == first
        if phase.name in skip or not (forced or _must_run(phase, found, state)):
            skipped.append(phase.name)
            continue

        # What the phase marks used is what it applied; take_used first drops what an earlier phase left.
        pc.decisions.take_used()
        pc.reuse_draft = _only_decisions_changed(state["phases"].get(phase.name), found)
        counts = phase.run(pc)
        ran.append(phase.name)
        cache["hits"] += counts.get("cache_hits", 0)
        cache["misses"] += counts.get("cache_misses", 0)
        record(state, phase, found, pc.decisions.take_used())
        save_state(pc.out_dir, state)
        if review is Review.STOP and phase.review_point:
            paused = phase.name
            break

    return {"ran": ran, "skipped": skipped, "paused_after": paused, "llm": cache}


def _finish(pc: PhaseContext, state: dict, paused: str | None) -> None:
    # Saves the pause, then rewrites the REVIEW.md frame and the explorer from the final state.
    state.pop(PAUSED_AFTER, None)
    state.pop(HUMAN_REVIEW, None)
    if paused:
        state[PAUSED_AFTER] = paused
        state[HUMAN_REVIEW] = True
    save_state(pc.out_dir, state)
    write_review_frame(pc, state)
    _write_explorer(pc, paused)


def _status(phase: Phase, pc: PhaseContext, saved: dict | None, states: dict[str, str]) -> tuple[str, str]:
    if saved is None:
        return _NEVER_BUILT, _NEVER_BUILT

    reason = changed_reason(saved["parts"], parts(phase, pc))
    if reason:
        return _STALE, reason
    if any(states[name] != _UP_TO_DATE for name in _upstream(phase)):
        return _STALE, _UPSTREAM_STALE
    return _UP_TO_DATE, ""


def _upstream(phase: Phase) -> list[str]:
    """Names of the phases that write a file `phase` reads."""
    return [other.name for other in PHASES if set(other.outputs) & set(phase.inputs)]


def _must_run(phase: Phase, found: dict[str, str], state: dict) -> bool:
    # Upstream changes show in the hash of the files this phase reads.
    saved = state["phases"].get(phase.name)
    return saved is None or saved["parts"] != found


def _only_decisions_changed(saved: dict | None, found: dict[str, str]) -> bool:
    if saved is None or saved["parts"] == found:
        return False
    return {key: value for key, value in saved["parts"].items() if key != "decisions"} == {
        key: value for key, value in found.items() if key != "decisions"
    }


def _range(start: str | None, stop: str | None) -> tuple[int, int]:
    names = [phase.name for phase in PHASES]
    for name in (start, stop):
        if name is not None and name not in names:
            raise ValueError(f"unknown phase: {name}; phases are {', '.join(names)}")

    first = names.index(start) if start else 0
    last = names.index(stop) if stop else len(names) - 1
    if first > last:
        raise ValueError(f"--from {start} comes after --to {stop}")
    return first, last


def _move_old_event_types(pc: PhaseContext) -> None:
    # A pre-plan build kept event types in ontology.yaml; this moves them to event_types.yaml, with no model call.
    path = pc.out_dir / "ontology.yaml"
    if not path.is_file() or (pc.out_dir / "event_types.yaml").is_file():
        return
    if read_yaml(path).get("event_types"):
        run_event_types(pc)


def _restamp_ontology(pc: PhaseContext) -> None:
    if not (pc.out_dir / "ontology.yaml").is_file():
        return
    roles = read_yaml(pc.out_dir / "roles.yaml")
    restamp_legacy_version(pc.out_dir, roles, read_json(pc.out_dir / "profile.json"), pc.context)


def _write_explorer(pc: PhaseContext, paused: str | None) -> None:
    # Imported here so a command that only reads status never loads the explorer.
    from ctxlayer.serving.explorer import write_explorer

    write_explorer(pc.settings, state={"paused_after": paused, "phases": phase_status(pc)})
