"""Scripted LLM role replies for the domain_a tables, and a helper that builds (profile, roles) with them."""
import json
import re
from pathlib import Path

from fakes import FakeLLM

from ctxlayer.common.config import Settings
from ctxlayer.common.context_files import load_context_files
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.profile import profile_database
from ctxlayer.common.roles import detect_roles

_TABLE_LINE = re.compile(r"^Table: (\S+)$", re.MULTILINE)


def _col(role: str, name: str, description: str, confidence: float = 0.95, **extra) -> dict:
    return {"role": role, "logical_name": name, "description": description, "confidence": confidence, **extra}


ROLES_REPLIES: dict[str, dict] = {
    "grain_quotes": {
        "kind": "measures",
        "primary_measure": "price_mid",
        "scenario_values": {"observed": "actual", "outlook": "projected"},
        "columns": {
            "quote_id": _col("row_id", "quote_id", "Surrogate key of the quote row."),
            "series_code": _col("entity_id", "series_code", "Code of the price series."),
            "series_title": _col("entity_label", "series_title", "Title of the price series."),
            "crop": _col("hierarchy_level", "crop", "Crop, the lowest hierarchy level."),
            "crop_group": _col("hierarchy_level", "crop_group", "Group the crop belongs to."),
            "market_region": _col("location", "market_region", "Region where the price is quoted."),
            "price_low": _col("measure_band", "price_low", "Low end of the price band.", band="low", additive=False),
            "price_mid": _col("measure_band", "price_mid", "Mid price of the band.", band="mid", additive=False),
            "price_high": _col("measure_band", "price_high", "High end of the price band.", band="high", additive=False),
            "period_ym": _col("period_key", "period_ym", "Quote month as YYYYMM."),
            "cadence": _col("frequency", "cadence", "Weekly or monthly sampling."),
            "currency": _col("currency", "currency", "Currency of the prices."),
            "unit": _col("unit", "unit", "Unit the prices are quoted per."),
            "quote_date": _col("time_business", "quote_date", "Date of the quote."),
            "revised_date": _col("time_other", "revised_date", "Date the quote was last revised.",
                                 meaning="last revision of the quote"),
            "basis": _col("scenario", "basis", "Observed or outlook (projected) value."),
            "loaded_at": _col("audit", "loaded_at", "Load timestamp."),
            "loaded_by": _col("audit", "loaded_by", "Loading process."),
        },
    },
    "farm_bulletins": {
        "kind": "text",
        "columns": {
            "row_id": _col("row_id", "row_id", "Surrogate key of the row."),
            "bulletin_ref": _col("doc_id", "bulletin_ref", "Reference of the bulletin."),
            "crop_tag": _col("doc_tag", "crop_tag", "Crop the bulletin is tagged with."),
            "region_tag": _col("doc_tag", "region_tag", "Region the bulletin is tagged with."),
            "headline": _col("text_title", "headline", "Bulletin headline."),
            "released_at": _col("time_business", "released_at", "Release time of the bulletin."),
            "digest": _col("text_summary", "digest", "Short summary of the bulletin."),
            "body_part_1": _col("text_part", "body_part_1", "First part of the body.", part_order=1),
            "body_part_2": _col("text_part", "body_part_2", "Second part of the body.", part_order=2),
            "body_part_3": _col("text_part", "body_part_3", "Third part of the body.", part_order=3),
            "outlet": _col("dimension", "outlet", "Publication that issued the bulletin."),
            "ingested_at": _col("audit", "ingested_at", "Ingestion timestamp."),
        },
    },
}


def _reply_for_table(prompt: str, tools) -> str:
    table = _TABLE_LINE.search(prompt).group(1)
    return json.dumps(ROLES_REPLIES[table])


def build_roles(db_path: Path, settings: Settings) -> tuple[dict, dict]:
    """(profile, roles) of the database at `db_path`, with the scripted replies standing in for the LLM."""
    con = connect_readonly(db_path)
    try:
        profile = profile_database(con, settings)
    finally:
        con.close()

    llm = FakeLLM({"roles": _reply_for_table})
    context = load_context_files(settings, db_path.stem)
    return profile, detect_roles(profile, llm, context, settings)
