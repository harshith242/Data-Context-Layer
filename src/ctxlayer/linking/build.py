"""Links build: runs the links phase."""
from ctxlayer.common.config import Settings
from ctxlayer.common.phase_context import make_context
from ctxlayer.linking.phases import run_links


def build_links(settings: Settings) -> dict:
    """Write links/candidate_drivers.jsonl, links/nodes.jsonl (Move nodes) and links/edges.jsonl; returns counts only."""
    return run_links(make_context(settings))
