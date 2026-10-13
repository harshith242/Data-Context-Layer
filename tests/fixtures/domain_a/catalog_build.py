"""Scripted LLM replies for the domain_a ontology and catalog, and a helper that builds both with them."""
import json
from pathlib import Path

from fakes import FakeLLM

from ctxlayer.common.catalog import Entity, seed_catalog
from ctxlayer.common.config import Settings
from ctxlayer.common.context_files import load_context_files
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.ontology import draft_ontology

ONTOLOGY_REPLY = {
    "types": {
        "grain_quotes.series_code": {"name": "Series", "description": "A published price series."},
        "grain_quotes.crop": {"name": "Crop", "description": "A traded crop."},
        "grain_quotes.crop_group": {"name": "CropGroup", "description": "A family of crops."},
    }
}

# Alias spellings match the bulletin tags in make_domain.py, which differ from the quotes table.
ALIASES_REPLY = {
    "aliases": {
        "crop:milling_wheat": ["Wheat (milling)"],
        "crop:malting_barley": ["Barley (malting)"],
        "crop:rapeseed": ["Rapeseed (canola)"],
        "crop:maize": ["Corn"],
        "crop:soybeans": ["Soy"],
        "crop:oats": ["Oats (spring)"],
        "location:great_plains": ["GP"],
        "location:rhine_valley": ["Rhine Vly"],
        "location:prairie_basin": ["Prairie Bsn"],
    }
}

PARENTS_REPLY = {
    "parents": {
        "location:great_plains": {"name": "North America", "confidence": 0.9},
        "location:prairie_basin": {"name": "North America", "confidence": 0.9},
        "location:rhine_valley": {"name": "Europe", "confidence": 0.9},
    }
}


def build_catalog(db_path: Path, profile: dict, roles: dict, settings: Settings) -> tuple[dict, list[Entity]]:
    """(ontology, entities) of the database at `db_path`, with the scripted replies standing in for the LLM."""
    llm = FakeLLM({
        "ontology": json.dumps(ONTOLOGY_REPLY),
        "aliases": json.dumps(ALIASES_REPLY),
        "location_parents": json.dumps(PARENTS_REPLY),
    })
    context = load_context_files(settings, db_path.stem)
    ontology = draft_ontology(roles, profile, llm, context)

    con = connect_readonly(db_path)
    try:
        return ontology, seed_catalog(con, roles, ontology, llm, settings)
    finally:
        con.close()
