"""Tests for the ontology draft, with a scripted LLM on the synthetic domain_a database."""
import copy
import json
import re

from ctxlayer.common.context_files import ContextFiles
from ctxlayer.common.ontology import BUILTIN_TYPES, draft_ontology
from fakes import FakeLLM
from fixtures.domain_a.catalog_build import ONTOLOGY_REPLY

_NO_CONTEXT = ContextFiles()


def _draft(domain_a_roles, reply, context=_NO_CONTEXT):
    profile, roles = domain_a_roles
    llm = FakeLLM({"ontology": reply if isinstance(reply, str) else json.dumps(reply)})
    return draft_ontology(roles, profile, llm, context)


def _types(ontology) -> dict:
    return {item["name"]: item for item in ontology["types"]}


def _relations(ontology, name) -> set[tuple]:
    return {(item["from"], item["to"]) for item in ontology["relations"] if item["name"] == name}


def test_induced_types_are_named_by_the_llm_beside_builtins(domain_a_roles):
    types = _types(_draft(domain_a_roles, ONTOLOGY_REPLY))

    assert set(types) == set(BUILTIN_TYPES) | {"Series", "Crop", "CropGroup"}
    assert types["Series"]["sources"] == [{"table": "grain_quotes", "columns": ["series_code", "series_title"]}]
    assert types["Crop"]["sources"] == [{"table": "grain_quotes", "columns": ["crop"]}]
    assert types["Series"]["description"] == "A published price series."




def test_has_parent_runs_from_child_to_parent_as_in_the_fds(domain_a_roles):
    profile, _ = domain_a_roles
    assert ["crop", "crop_group"] in profile["tables"]["grain_quotes"]["fds"]

    relations = _draft(domain_a_roles, ONTOLOGY_REPLY)["relations"]
    parents = {(r["from"], r["to"], r["table"]) for r in relations if r["name"] == "HAS_PARENT"}

    assert parents == {("Crop", "CropGroup", "grain_quotes"), ("Series", "CropGroup", "grain_quotes")}






def test_version_is_a_hash_of_inputs_not_of_llm_output(domain_a_roles):
    first = _draft(domain_a_roles, ONTOLOGY_REPLY)
    other_names = copy.deepcopy(ONTOLOGY_REPLY)
    other_names["types"]["grain_quotes.crop"]["name"] = "Commodity"
    second = _draft(domain_a_roles, other_names)
    third = _draft(domain_a_roles, ONTOLOGY_REPLY, ContextFiles(database="Prices are spot prices."))

    assert re.fullmatch(r"[0-9a-f]{64}", first["version"])
    assert second["version"] == first["version"]
    assert third["version"] != first["version"]


def test_malformed_reply_falls_back_to_names_from_columns(domain_a_roles):
    types = _types(_draft(domain_a_roles, "not json at all"))

    assert set(types) == set(BUILTIN_TYPES) | {"SeriesCode", "Crop", "CropGroup"}


def test_invalid_name_falls_back_only_for_that_group(domain_a_roles):
    reply = copy.deepcopy(ONTOLOGY_REPLY)
    reply["types"]["grain_quotes.series_code"]["name"] = "PriceSeries"
    reply["types"]["grain_quotes.crop"]["name"] = "Document"  # builtin name
    reply["types"]["grain_quotes.crop_group"]["name"] = "crop group"  # not PascalCase
    types = _types(_draft(domain_a_roles, reply))

    assert set(types) == set(BUILTIN_TYPES) | {"PriceSeries", "Crop", "CropGroup"}
    assert types["PriceSeries"]["description"] == "A published price series."


def test_version_ignores_headings_the_ontology_step_does_not_read(domain_a_roles):
    def version(text: str) -> str:
        return _draft(domain_a_roles, ONTOLOGY_REPLY, ContextFiles(database=text))["version"]

    base = "Intro.\n## Names\nGP is the Great Plains.\n## News\n"

    assert version(base + "Strikes matter.") == version(base + "Weather matters.")
    assert version(base) != version(base.replace("Great Plains", "Prairie Basin"))
