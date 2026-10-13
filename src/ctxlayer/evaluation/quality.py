"""Context-layer quality counts: role accuracy, tag resolution, extraction spot-check sheet."""
import json
import random
from pathlib import Path

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.files import read_jsonl, read_yaml

_SPOTCHECK_DOCS = 30
_SPOTCHECK_SEED = 0
_TAGGED_WITH = "TAGGED_WITH"


def quality_checks(settings: Settings, spotcheck_path: Path | None = None) -> dict:
    """Counts only; a check whose input file is absent is left out of the result."""
    artifacts = artifacts_dir(settings)
    eval_dir = settings.path("paths.eval")
    spotcheck_path = spotcheck_path or eval_dir / "extraction_spotcheck.md"
    counts: dict = {}

    counts.update(_role_accuracy(artifacts / "roles.yaml", eval_dir / "role_labels.yaml"))
    counts.update(_tag_resolution(artifacts / "graph"))
    counts.update(_spotcheck(artifacts / "docs" / "extractions.jsonl", spotcheck_path))

    return counts


def _role_accuracy(roles_path: Path, labels_path: Path) -> dict:
    if not (roles_path.is_file() and labels_path.is_file()):
        return {}

    # labels: {table: {column: role}}; a column missing from roles.yaml counts as wrong.
    tables = read_yaml(roles_path).get("tables", {})
    correct = total = 0
    for table, columns in (read_yaml(labels_path) or {}).items():
        detected = tables.get(table, {}).get("columns", {})
        for column, role in columns.items():
            total += 1
            correct += detected.get(column, {}).get("role") == role

    return {"role_correct": correct, "role_total": total}


def _tag_resolution(graph_dir: Path) -> dict:
    counts = {}

    entities_path = graph_dir / "entities_text.yaml"
    if entities_path.is_file():
        entities = read_yaml(entities_path) or []
        counts["tag_new_entities"] = len(entities)
        counts["tag_new_flagged"] = sum(bool(entity.get("needs_review")) for entity in entities)

    edges_path = graph_dir / "edges.jsonl"
    if edges_path.is_file():
        edges = [edge for edge in read_jsonl(edges_path) if edge.get("rel") == _TAGGED_WITH]
        counts["tag_edges"] = len(edges)
        counts["tag_edges_flagged"] = sum(bool(edge.get("needs_review")) for edge in edges)

    return counts


def _spotcheck(extractions_path: Path, out_path: Path) -> dict:
    if not extractions_path.is_file():
        return {}

    # Seeded sample so the sheet is the same on every run.
    records = read_jsonl(extractions_path)
    sample = random.Random(_SPOTCHECK_SEED).sample(records, min(_SPOTCHECK_DOCS, len(records)))

    lines = ["# Extraction spot-check", "", "Mark each event correct or wrong against its document.", ""]
    for record in sample:
        lines.append(f"## {record['doc_id']}")
        lines.extend(f"- {json.dumps(event, ensure_ascii=False)}" for event in record.get("events", []))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    return {"spotcheck_docs": len(sample)}

