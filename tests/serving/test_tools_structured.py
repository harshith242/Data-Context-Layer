"""Context and structured tools over artifacts written from the synthetic domain."""
import pytest
from fakes import FakeEmbedder, FakeLLM
from fixtures.domain_a.make_domain import PLANTED_SERIES_CODE

from ctxlayer.common.catalog import save_catalog
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import write_yaml
from ctxlayer.serving.context import get_context
from ctxlayer.serving.tools import build_tools, open_context

_JUDGE_SAYS_NONE = FakeLLM({"resolve_judge": '{"id": null, "confidence": 0}'})
_QUERY = "latest_value"


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts the passages it embeds."""

    def __init__(self):
        super().__init__()
        self.passages = 0

    def embed_passages(self, texts):
        self.passages += len(texts)
        return super().embed_passages(texts)

    def embed_queries(self, texts):
        return super().embed_passages(texts)


@pytest.fixture
def make_ctx(domain_a_semantic, domain_a_catalog):
    settings, _ = domain_a_semantic
    ontology, entities = domain_a_catalog
    save_catalog(entities, artifacts_dir(settings) / "entities.yaml")
    write_yaml(artifacts_dir(settings) / "ontology.yaml", ontology)

    opened = []

    def make(llm=None, embedder=None):
        ctx = open_context(settings, embedder or FakeEmbedder(), llm)
        opened.append(ctx)
        return ctx

    yield make
    for ctx in opened:
        ctx.con.close()


def _tool(ctx, name: str):
    return next(tool for tool in build_tools(ctx) if tool.name == name)


def _planted_typo(ctx) -> tuple[str, str]:
    """(canonical name, name with its last letter replaced) of the planted series."""
    name = next(e.canonical_name for e in ctx.catalog if PLANTED_SERIES_CODE in e.aliases)
    return name, name[:-1] + ("x" if name[-1] != "x" else "y")


def test_query_metric_resolves_a_typo_to_the_same_rows_as_the_exact_name(make_ctx):
    ctx = make_ctx(_JUDGE_SAYS_NONE)
    name, typo = _planted_typo(ctx)

    fuzzy = _tool(ctx, "query_metric").fn(metric=_QUERY, entities=[typo])
    exact = _tool(ctx, "query_metric").fn(metric=_QUERY, entities=[name])

    assert fuzzy["rows"] == exact["rows"]
    assert fuzzy["rows"][0][fuzzy["columns"].index("series_code")] == PLANTED_SERIES_CODE


def test_get_context_embeds_the_corpus_once_and_only_the_question_after(make_ctx):
    embedder = _CountingEmbedder()
    ctx = make_ctx(embedder=embedder)

    get_context(ctx, "price of oats")
    corpus = embedder.passages
    get_context(ctx, "price of maize")

    assert corpus > 0
    assert embedder.passages == corpus


def test_get_context_puts_a_quoted_alias_first_via_the_resolver(make_ctx):
    ctx = make_ctx()

    first = get_context(ctx, "How is 'Wheat (milling)' doing?")["entities"][0]

    assert (first["id"], first["method"]) == ("crop:milling_wheat", "alias")


def test_get_context_makes_no_llm_call(make_ctx):
    ctx = make_ctx(_JUDGE_SAYS_NONE)
    _, typo = _planted_typo(ctx)

    get_context(ctx, f'Show "{typo}"')

    # Span resolution is exact + alias only; the judge never runs inside get_context.
    assert ctx.llm.calls == []
