"""The two paid caches (extractions, judge verdicts) and the chunk index: robust to cut-short files."""
import json

import pytest
from fakes import FakeEmbedder, FakeLLM
from fixtures.domain_a.extract_replies import EVENT_TYPES_REPLY, extract_reply
from fixtures.domain_a.structured_build import structured_llm
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.files import read_yaml, write_json
from ctxlayer.common.resolve import Cascade, Resolver
from ctxlayer.structured.build import build_structured
from ctxlayer.unstructured.build import build_unstructured
from ctxlayer.unstructured.chunk import Chunk
from ctxlayer.unstructured.documents import build_documents
from ctxlayer.unstructured.embed import index_chunks
from ctxlayer.unstructured.extract import _cache_key, _prompt, extract_all

_TYPES = ["Document", "Event", "Organization", "Location", "Move", "Series", "Crop", "CropGroup"]
_NEW_MENTION = "northern plains area"
_OTHER_MENTION = "eastern coast area"
_TASK_EXTRACT = "[task: extract]"
_TASK_BATCH = "[task: resolve_judge_batch]"


class _NamedLLM(FakeLLM):
    """A FakeLLM that reports which model it stands for."""

    def __init__(self, model_name: str, handlers: dict):
        super().__init__(handlers)
        self.model_name = model_name


def _count(llm: FakeLLM, tag: str) -> int:
    return sum(1 for prompt, _ in llm.calls if prompt.startswith(tag))


@pytest.fixture(scope="module")
def docs(domain_a, domain_a_built):
    roles = read_yaml(artifacts_dir(domain_a_built) / "roles.yaml")
    con = connect_readonly(domain_a.db_path)
    try:
        return build_documents(con, roles)
    finally:
        con.close()


@pytest.fixture
def ontology() -> dict:
    other = {"name": "other", "description": "None fit."}
    return {"types": [{"name": name} for name in _TYPES], "event_types": EVENT_TYPES_REPLY["event_types"] + [other]}


# ------------------------------------------------------------ atomic writes

def _circular() -> list:
    loop: list = []
    loop.append(loop)
    return loop


def test_failed_serialisation_leaves_the_old_file_and_no_temp_file(tmp_path):
    target = tmp_path / "out" / "data.txt"
    target.parent.mkdir()
    target.write_text("old content")

    with pytest.raises(Exception):
        write_json(target, _circular())

    assert target.read_text() == "old content"
    assert [path.name for path in target.parent.iterdir()] == ["data.txt"]




# --------------------------------------------------------- extraction cache

def test_cut_short_extraction_entry_is_extracted_again_and_rewritten_whole(docs, ontology, tmp_path):
    cache = tmp_path / "cache"
    first, _ = extract_all(docs, FakeLLM({"extract": extract_reply}), ontology, cache, workers=1)
    damaged = sorted(cache.glob("*.json"))[0]
    whole = damaged.read_bytes()
    damaged.write_bytes(whole[: len(whole) // 2])

    llm = FakeLLM({"extract": extract_reply})
    second, failures = extract_all(docs, llm, ontology, cache, workers=1)

    assert _count(llm, _TASK_EXTRACT) == 1
    assert failures == [] and second == first
    assert damaged.read_bytes() == whole


def test_two_model_names_use_two_cache_files_for_the_same_document(docs, ontology, tmp_path):
    cache = tmp_path / "cache"
    first = _NamedLLM("model-a", {"extract": extract_reply})
    second = _NamedLLM("model-b", {"extract": extract_reply})

    extract_all(docs, first, ontology, cache, workers=1)
    extract_all(docs, second, ontology, cache, workers=1)

    assert _count(second, _TASK_EXTRACT) == len(docs)
    assert len(list(cache.glob("*.json"))) == 2 * len(docs)




def test_entry_under_the_old_key_is_used_and_copied_to_the_new_key(docs, ontology, tmp_path):
    cache = tmp_path / "cache"
    first, _ = extract_all(docs, FakeLLM({"extract": extract_reply}), ontology, cache, workers=1)
    types = [item["name"] for item in ontology["types"]]
    for doc in docs:
        prompt = _prompt(doc, ontology["event_types"], types, "")
        (cache / f"{_cache_key(prompt, 'default')}.json").rename(cache / f"{_cache_key(prompt)}.json")

    llm = _NamedLLM("model-a", {"extract": extract_reply})
    second, failures = extract_all(docs, llm, ontology, cache, workers=1)

    assert _count(llm, _TASK_EXTRACT) == 0
    assert failures == [] and second == first
    new_files = {cache / f"{_cache_key(_prompt(doc, ontology['event_types'], types, ''), 'model-a')}.json" for doc in docs}
    assert new_files <= set(cache.glob("*.json"))


# ------------------------------------------------------------ judge verdicts

def _entities(domain_a_catalog):
    return domain_a_catalog[1]


def _resolver(settings, entities, llm, log_dir) -> Resolver:
    return Resolver(
        entities, FakeEmbedder(), llm, settings, Cascade.FULL, cache_path=log_dir / "entity_embeddings.npy",
    )


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def _judge_llm() -> FakeLLM:
    return FakeLLM({"resolve_judge_batch": "{}"})


def test_cut_short_last_verdict_line_is_skipped_and_the_rest_loads(project_settings, domain_a_catalog, tmp_path):
    entities = _entities(domain_a_catalog)
    first = _judge_llm()
    _resolver(project_settings, entities, first, tmp_path).prefetch(
        [(_NEW_MENTION, "Location"), (_OTHER_MENTION, "Location")],
    )
    log = tmp_path / "resolve_judge.jsonl"
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    log.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2])

    second = _judge_llm()
    resolver = _resolver(project_settings, entities, second, tmp_path)
    resolver.prefetch([(_NEW_MENTION, "Location"), (_OTHER_MENTION, "Location")])

    # Only the cut-short mention is asked again, and the log stays readable line by line.
    assert _count(second, _TASK_BATCH) == 1
    assert _OTHER_MENTION in second.calls[0][0] and _NEW_MENTION not in second.calls[0][0]
    readable = [line for line in log.read_text().splitlines() if _is_json(line)]
    assert len(readable) == 2  # the first verdict and the one just re-asked, on its own line


def test_verdicts_survive_an_alias_added_to_an_unrelated_entity(project_settings, domain_a_catalog, tmp_path):
    entities = _entities(domain_a_catalog)
    first = _judge_llm()
    _resolver(project_settings, entities, first, tmp_path).prefetch([(_NEW_MENTION, "Location")])
    unrelated = next(entity for entity in entities if entity.type == "Crop")
    unrelated.aliases = [*unrelated.aliases, "brand new alias"]

    try:
        second = _judge_llm()
        _resolver(project_settings, entities, second, tmp_path).prefetch([(_NEW_MENTION, "Location")])
    finally:
        unrelated.aliases.remove("brand new alias")

    assert _count(first, _TASK_BATCH) == 1
    assert _count(second, _TASK_BATCH) == 0


# --------------------------------------------------------------- chunk index

class _CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.embedded = 0

    def embed_passages(self, texts):
        self.embedded += len(texts)
        return super().embed_passages(texts)


def test_cut_short_chunks_file_means_everything_is_embedded_again(tmp_path):
    chunks = [
        Chunk("d1#1", "d1", "2025-01-01", "first text", "body"),
        Chunk("d2#1", "d2", "2025-01-01", "second text", "body"),
    ]
    index_chunks(chunks, FakeEmbedder(), tmp_path)
    path = tmp_path / "chunks.parquet"
    path.write_bytes(path.read_bytes()[:50])

    embedder = _CountingEmbedder()
    counts = index_chunks(chunks, embedder, tmp_path)

    assert embedder.embedded == len(chunks)
    assert counts["chunks"] == len(chunks)

    reuse = _CountingEmbedder()
    index_chunks(chunks, reuse, tmp_path)
    assert reuse.embedded == 0  # the rewritten file is whole


# ------------------------------------------------------ correctness, no cache

_CACHES = ("docs/extract_cache", "docs/chunks.parquet", "resolve_judge.jsonl", "entity_embeddings.npy",
           "entity_embeddings.npy.sha256")
_GRAPH_FILES = ("graph/nodes.jsonl", "graph/edges.jsonl", "docs/extractions.jsonl")


def test_build_without_any_cache_gives_the_same_graph_as_with_warm_caches(tmp_path_factory, domain_a, domain_a_built):
    settings = copy_built(tmp_path_factory.mktemp("no_cache"), domain_a, domain_a_built)
    out = artifacts_dir(settings)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    warm_llm = unstructured_llm()
    build_unstructured(settings, warm_llm, FakeEmbedder())
    warm = {name: (out / name).read_bytes() for name in _GRAPH_FILES}

    for name in _CACHES:
        path = out / name
        if path.is_dir():
            for entry in path.iterdir():
                entry.unlink()
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
    cold_llm = unstructured_llm()
    build_unstructured(settings, cold_llm, FakeEmbedder())

    assert _count(warm_llm, _TASK_EXTRACT) == 0
    assert _count(cold_llm, _TASK_EXTRACT) > 0
    assert {name: (out / name).read_bytes() for name in _GRAPH_FILES} == warm
