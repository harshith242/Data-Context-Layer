"""Tests for the context.md loader."""
import pytest
from fakes import FakeEmbedder
from fixtures.domain_a.structured_build import structured_llm
from fixtures.domain_a.unstructured_build import unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.context_files import Step, load_context_files
from ctxlayer.common.files import read_jsonl
from ctxlayer.common.ontology import load_ontology
from ctxlayer.structured.build import build_structured
from ctxlayer.unstructured.build import build_unstructured
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.extract import _prompt

_DB = "domain_a"


@pytest.fixture
def sources(tmp_settings):
    """Settings factory plus a sources folder holding a database file and one table file."""

    def make():
        settings = tmp_settings()
        folder = settings.path("paths.sources") / _DB
        folder.mkdir(parents=True)
        (folder / "context.md").write_text("Prices are quoted in the listed currency.\n")
        (folder / "grain_quotes.context.md").write_text("basis marks outlook rows.\n")
        return settings

    return make


def test_existing_files_are_always_loaded_for_database_and_tables(sources):
    context = load_context_files(sources(), _DB)

    assert context.database == "Prices are quoted in the listed currency."
    assert context.tables == {"grain_quotes": "basis marks outlook rows."}








_HEADED = (
    "Intro line.\n\n## Columns\nCOLUMNS-HINT\n\n## Names\nNAMES-HINT\n\n"
    "## Currencies\nCURRENCY-HINT\n\n## News\nNEWS-HINT\n"
)


def _write_context(settings, text: str) -> None:
    folder = settings.path("paths.sources") / _DB
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "context.md").write_text(text)


def test_for_step_returns_the_preamble_and_only_the_steps_headings(tmp_settings):
    settings = tmp_settings()
    _write_context(settings, _HEADED)

    text = load_context_files(settings, _DB).for_step(Step.CATALOG)

    assert text == "Intro line.\n\n## Names\nNAMES-HINT"


@pytest.mark.parametrize("step, expected", [
    (Step.ROLES, {"COLUMNS-HINT"}),
    (Step.EVENTS, {"NEWS-HINT", "NAMES-HINT"}),
])
def test_each_step_reads_its_headings_from_the_spec_table(tmp_settings, step, expected):
    settings = tmp_settings()
    _write_context(settings, _HEADED)

    text = load_context_files(settings, _DB).for_step(step)

    found = {hint for hint in ("COLUMNS-HINT", "NAMES-HINT", "CURRENCY-HINT", "NEWS-HINT") if hint in text}
    assert found == expected
    assert "Intro line." in text




def test_file_with_only_unknown_headings_goes_whole_to_every_step(tmp_settings):
    settings = tmp_settings()
    text = "## Notes\nfirst\n\n## Other\nsecond"
    _write_context(settings, text)

    context = load_context_files(settings, _DB)

    assert [context.for_step(step) for step in Step] == [text] * len(Step)


def test_unknown_heading_after_a_known_one_counts_as_preamble(tmp_settings):
    settings = tmp_settings()
    _write_context(settings, "## Names\nNAMES-HINT\n\n## Other\nOTHER-HINT")

    context = load_context_files(settings, _DB)

    assert "OTHER-HINT" in context.for_step(Step.ROLES)
    assert "NAMES-HINT" not in context.for_step(Step.ROLES)




def test_table_file_is_split_by_heading_and_only_added_for_its_table(sources):
    settings = sources()
    folder = settings.path("paths.sources") / _DB
    (folder / "grain_quotes.context.md").write_text("## Columns\nTABLE-COLS\n\n## News\nTABLE-NEWS")
    context = load_context_files(settings, _DB)

    roles = context.for_step(Step.ROLES, "grain_quotes")

    assert roles == "Prices are quoted in the listed currency.\n\n## Columns\nTABLE-COLS"
    assert context.for_step(Step.ROLES, "farm_bulletins") == "Prices are quoted in the listed currency."
    assert "TABLE-NEWS" in context.for_step(Step.EVENTS)




def test_max_chars_comes_from_settings(tmp_settings):
    settings = tmp_settings({"context": {"max_chars": 25}})
    _write_context(settings, "y" * 100)

    text = load_context_files(settings, _DB).for_step(Step.AGENT)

    assert text.startswith("y" * 25) and text.endswith("(cut)")


def test_names_line_reaches_aliases_and_judge_prompts_but_not_roles(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    folder = settings.path("paths.sources") / domain_a.db_path.stem
    folder.mkdir(parents=True)
    (folder / "context.md").write_text("## Names\nNAMES-HINT\n\n## News\nNEWS-HINT\n")
    structured, unstructured = structured_llm(), unstructured_llm()

    build_structured(settings, structured, FakeEmbedder())
    build_unstructured(settings, unstructured, FakeEmbedder())

    def with_hint(llm, task: str, hint: str) -> list[bool]:
        return [hint in prompt for prompt, _ in llm.calls if prompt.startswith(f"[task: {task}]")]

    assert with_hint(structured, "aliases", "NAMES-HINT") == [True]
    assert with_hint(structured, "location_parents", "NAMES-HINT") == [True]
    judge = with_hint(unstructured, "resolve_judge_batch", "NAMES-HINT")
    assert judge and all(judge)
    assert not any(with_hint(structured, "roles", "NAMES-HINT"))
    assert not any(with_hint(unstructured, "tag_types", "NAMES-HINT"))
    assert not any(with_hint(structured, "aliases", "NEWS-HINT"))
    assert all(with_hint(unstructured, "extract", "NEWS-HINT"))


def test_without_context_the_extract_prompt_is_unchanged(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    unstructured = unstructured_llm()
    build_structured(settings, structured_llm(), FakeEmbedder())
    build_unstructured(settings, unstructured, FakeEmbedder())

    # The cache key is the hash of this prompt, so any drift would re-pay every extraction.
    out_dir = artifacts_dir(settings)
    ontology = load_ontology(out_dir)
    types = [item["name"] for item in ontology["types"]]
    docs = [Document(**row) for row in read_jsonl(out_dir / "docs" / "documents.jsonl")]
    expected = {_prompt(doc, ontology["event_types"], types, "") for doc in docs}
    actual = {prompt for prompt, _ in unstructured.calls if prompt.startswith("[task: extract]")}

    assert actual == expected
