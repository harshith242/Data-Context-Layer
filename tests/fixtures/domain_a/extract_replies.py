"""Scripted LLM replies for event-type induction and extraction over the domain_a bulletins."""
import json

from fixtures.domain_a.make_domain import PLANTED_QUOTE

EVENT_TYPES_REPLY = {
    "event_types": [
        {"name": "supply_disruption", "description": "Weather or logistics cut the available supply."},
        {"name": "policy_change", "description": "A government changes a tariff, quota or rule."},
        {"name": "demand_shift", "description": "Buyers change the volumes they want."},
        {"name": "trade_flow", "description": "Exports or imports change route or size."},
        {"name": "harvest_report", "description": "Crop estimates or harvest progress are published."},
        {"name": "price_move", "description": "A market price moves sharply."},
        {"name": "labour_action", "description": "Strikes or labour shortages hit operations."},
        {"name": "infrastructure_issue", "description": "Ports, rail or storage are disrupted."},
    ]
}

_FIRST_TITLE = "Drought cuts Great Plains wheat supply"
_SECOND_TITLE = "Dry weather shrinks Great Plains milling wheat crop"
_SECOND_QUOTE = "Dry weather across the Great Plains is shrinking the milling wheat crop"
_SECOND_AFFECT_QUOTE = "expected shortfall in domestic milling supply"
_REGION = "Great Plains"
_CROP_TAG = "Wheat (milling)"


def sentence_id(prompt: str, text: str) -> list[int]:
    """Number of the [S<n>] sentence that contains `text` in the numbered prompt."""
    for line in prompt.splitlines():
        if line.startswith("[S") and text in line:
            return [int(line[2:line.index("]")])]
    return []


def _drought_event(prompt: str, date: str, evidence: str, affect_quote: str) -> dict:
    return {
        "type": "supply_disruption",
        "date": date,
        "summary": "Drought cuts the Great Plains milling wheat crop.",
        "locations": [_REGION],
        "organizations": [],
        "affects": [{"entity": _CROP_TAG, "aspect": "price", "direction": "up", "confidence": 0.8,
                     "s": sentence_id(prompt, affect_quote)}],
        "s": sentence_id(prompt, evidence),
    }


def extract_reply(prompt: str, tools: list[dict] | None) -> str:
    """FakeLLM handler for `[task: extract]`: one drought event for each of the two planted bulletins."""
    events = []
    if _FIRST_TITLE in prompt:
        events.append(_drought_event(prompt, "2025-04-25", PLANTED_QUOTE, PLANTED_QUOTE))
    if _SECOND_TITLE in prompt:
        events.append(_drought_event(prompt, "2025-04-27", _SECOND_QUOTE, _SECOND_AFFECT_QUOTE))
    return json.dumps({"events": events}, separators=(",", ":"))
