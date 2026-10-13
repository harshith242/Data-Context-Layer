"""Scripted LLM for the whole domain_a build: structured tasks plus event types, extraction and tag types."""
import json
import shutil
from pathlib import Path

from fakes import FakeLLM
from fixtures.domain_a.extract_replies import EVENT_TYPES_REPLY, extract_reply
from fixtures.domain_a.make_domain import DomainA
from fixtures.domain_a.structured_build import STRUCTURED_HANDLERS

from ctxlayer.common.config import Settings, artifacts_dir, load_settings

TAG_TYPES_REPLY = {"tag_types": {"crop_tag": "Crop", "region_tag": "Location"}}
NO_MATCH_REPLY = {"id": None, "confidence": 0.0}

UNSTRUCTURED_HANDLERS = {
    **STRUCTURED_HANDLERS,
    "event_types": json.dumps(EVENT_TYPES_REPLY),
    "extract": extract_reply,
    "tag_types": json.dumps(TAG_TYPES_REPLY),
    "resolve_judge": json.dumps(NO_MATCH_REPLY),
    "resolve_judge_batch": "{}",  # every mention in a batch: no match
}


def unstructured_llm(**overrides) -> FakeLLM:
    """FakeLLM answering every build task of domain_a; `overrides` replace or add task handlers."""
    return FakeLLM({**UNSTRUCTURED_HANDLERS, **overrides})


def copy_built(root: Path, domain_a: DomainA, built: Settings) -> Settings:
    """Settings over a writable copy of the structured build `built`, under `root`."""
    (root / "config").mkdir()
    shutil.copy(built.root / "config" / "settings.yaml", root / "config" / "settings.yaml")
    overrides = {"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(root / "artifacts")}}
    settings = load_settings(root=root, overrides=overrides)
    shutil.copytree(artifacts_dir(built), artifacts_dir(settings))
    return settings
