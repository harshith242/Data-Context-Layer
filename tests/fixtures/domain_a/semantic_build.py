"""Scripted LLM replies for the domain_a semantic layer, and a helper that drafts and writes it."""
import json
from pathlib import Path

from fakes import FakeLLM

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.context_files import load_context_files
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.files import write_yaml
from ctxlayer.structured.semantic_draft import draft_semantic
from ctxlayer.structured.semantic_views import write_views

GRAINS_REPLY = {"weekly": "week", "monthly": "month"}
SYNONYMS_REPLY = {
    "price_mid": ["mid price", "average price"],
    "series_title": ["series name"],
    "crop": ["commodity"],
    "not_a_column": ["ignored"],
}


def scripted_llm() -> FakeLLM:
    return FakeLLM({
        "grains": json.dumps(GRAINS_REPLY),
        "conversions": "[]",
        "synonyms": json.dumps(SYNONYMS_REPLY),
    })


def build_semantic(db_path: Path, profile: dict, roles: dict, settings: Settings) -> dict:
    """Draft with scripted replies; write semantic.yaml and views.sql under the artifacts folder."""
    context = load_context_files(settings, db_path.stem)
    con = connect_readonly(db_path)
    try:
        semantic = draft_semantic(roles, profile, con, scripted_llm(), context)
    finally:
        con.close()

    write_yaml(artifacts_dir(settings) / "semantic.yaml", semantic)
    write_views(semantic, settings)
    return semantic
