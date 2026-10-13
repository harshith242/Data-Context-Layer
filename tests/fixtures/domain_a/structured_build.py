"""Scripted LLM for the whole structured build of domain_a (roles, ontology, catalog, semantic)."""
import json

from fakes import FakeLLM
from fixtures.domain_a.catalog_build import ALIASES_REPLY, ONTOLOGY_REPLY, PARENTS_REPLY
from fixtures.domain_a.roles_reply import _reply_for_table
from fixtures.domain_a.semantic_build import GRAINS_REPLY, SYNONYMS_REPLY

STRUCTURED_HANDLERS = {
    "roles": _reply_for_table,
    "ontology": json.dumps(ONTOLOGY_REPLY),
    "aliases": json.dumps(ALIASES_REPLY),
    "location_parents": json.dumps(PARENTS_REPLY),
    "grains": json.dumps(GRAINS_REPLY),
    "conversions": "[]",
    "synonyms": json.dumps(SYNONYMS_REPLY),
}


def structured_llm(**overrides) -> FakeLLM:
    """FakeLLM answering every structured-build task; `overrides` replace or add task handlers."""
    return FakeLLM({**STRUCTURED_HANDLERS, **overrides})
