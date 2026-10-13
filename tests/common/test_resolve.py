"""Tests for the entity resolver cascade: exact, alias, fuzzy, embedding + judge."""
import json
import numpy as np
import pytest

from ctxlayer.common.catalog import Entity
from ctxlayer.common.resolve import Cascade, Method, Resolver, Status
from fakes import FakeEmbedder, FakeLLM

_NO_ONE = '{"id": null, "confidence": 0.0}'


class _CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.passage_calls = 0

    def embed_passages(self, texts):
        self.passage_calls += 1
        return super().embed_passages(texts)

    def embed_queries(self, texts):
        # Queries must not count as passage embeddings.
        return FakeEmbedder.embed_passages(self, texts)


def _catalog() -> list[Entity]:
    return [
        Entity("p-sprite", "product", "Sprite Zero", aliases=["Sprite 0", "SPR-ZERO"]),
        Entity("p-diet", "product", "Diet Coke", aliases=["DC", "Coke Light"]),
        Entity("p-classic", "product", "Coca-Cola Classic", aliases=["Coke", "Cola"]),
        Entity("p-cola-zero", "product", "Cola Zero", aliases=["Cola"]),
        Entity("p-crush", "product", "Orange Crush", aliases=["Orange"]),
        Entity("l-tx", "location", "Texas", aliases=["TX"]),
        Entity("l-dal", "location", "Dallas", attributes={"parent": "l-tx"}),
        Entity("l-orange", "location", "Orange"),
    ]


def _resolver(
    tmp_settings,
    llm=None,
    cascade=Cascade.FULL,
    overrides=None,
    entities=None,
    embedder=None,
    cache_path=None,
) -> Resolver:
    return Resolver(
        entities if entities is not None else _catalog(),
        embedder or FakeEmbedder(),
        llm,
        tmp_settings(overrides),
        cascade=cascade,
        cache_path=cache_path,
    )


def _judge(reply: str) -> FakeLLM:
    return FakeLLM({"resolve_judge": reply})


def test_exact_canonical_name_matches_with_full_score(tmp_settings):
    result = _resolver(tmp_settings, _judge(_NO_ONE)).resolve("Diet Coke")

    assert result.status == Status.MATCHED
    [match] = result.matches
    assert (match.id, match.name, match.type) == ("p-diet", "Diet Coke", "product")
    assert (match.method, match.score, match.needs_review) == (Method.EXACT, 1.0, False)


def test_alias_matches_after_normalising_case_and_punctuation(tmp_settings):
    # "spr-zero" is not a canonical name; only the normalised alias "spr zero" can solve it.
    result = _resolver(tmp_settings, _judge(_NO_ONE)).resolve("  spr_ZERO ")

    [match] = result.matches
    assert (match.id, match.method, match.score) == ("p-sprite", Method.ALIAS, 1.0)


def test_alias_shared_by_two_entities_is_ambiguous(tmp_settings):
    result = _resolver(tmp_settings, _judge(_NO_ONE)).resolve("cola")

    assert result.status == Status.AMBIGUOUS
    assert [m.id for m in result.matches] == ["p-classic", "p-cola-zero"]
    assert {m.method for m in result.matches} == {Method.ALIAS}


def test_fuzzy_typo_matches_above_threshold(tmp_settings):
    # token_sort_ratio("sprite zeroo", "sprite zero") = 2*11/23 = 95.65
    result = _resolver(tmp_settings, _judge(_NO_ONE)).resolve("Sprite Zeroo")

    [match] = result.matches
    assert (match.id, match.method) == ("p-sprite", Method.FUZZY)
    assert match.score == pytest.approx(0.9565, abs=1e-4)
    assert match.needs_review is False




def test_fuzzy_score_below_review_threshold_needs_review(tmp_settings):
    # "sprite zer0" scores 90.91, under a 0.95 review threshold.
    resolver = _resolver(tmp_settings, _judge(_NO_ONE), overrides={"review": {"confidence_threshold": 0.95}})

    [match] = resolver.resolve("Sprite Zer0").matches

    assert match.score == pytest.approx(0.9091, abs=1e-4)
    assert match.needs_review is True


def test_embedding_and_judge_resolve_a_description(tmp_settings):
    # No exact, alias or fuzzy hit: only shared words with "Diet Coke, DC, Coke Light" help.
    llm = _judge('{"id": "p-diet", "confidence": 0.9}')

    result = _resolver(tmp_settings, llm).resolve("light coke soda")

    [match] = result.matches
    assert (match.id, match.method, match.score, match.needs_review) == ("p-diet", Method.EMBEDDING, 0.9, False)
    prompt = llm.calls[0][0]
    assert prompt.splitlines()[0] == "[task: resolve_judge]"
    assert "light coke soda" in prompt
    assert "p-diet" in prompt and "Diet Coke" in prompt


def test_judge_low_confidence_needs_review(tmp_settings):
    llm = _judge('{"id": "p-diet", "confidence": 0.6}')

    [match] = _resolver(tmp_settings, llm).resolve("light coke soda").matches

    assert (match.id, match.score, match.needs_review) == ("p-diet", 0.6, True)


def test_judge_answers_none_gives_no_match_with_fuzzy_suggestions(tmp_settings):
    # "sprite zeroxxx" scores 88 (< 90), so the judge decides, and refuses.
    result = _resolver(tmp_settings, _judge(_NO_ONE)).resolve("Sprite Zeroxxx")

    assert result.status == Status.NO_MATCH
    assert result.matches == []
    assert result.suggestions[0] == "Sprite Zero"


def test_judge_id_outside_candidates_counts_as_none(tmp_settings):
    # "l-tx" exists in the catalog but is not a product candidate for this lookup.
    llm = _judge('{"id": "l-tx", "confidence": 0.99}')

    result = _resolver(tmp_settings, llm).resolve("light coke soda", type="product")

    assert result.status == Status.NO_MATCH


def test_malformed_judge_reply_counts_as_none(tmp_settings):
    result = _resolver(tmp_settings, _judge("I think it is Diet Coke")).resolve("light coke soda")

    assert result.status == Status.NO_MATCH


def test_type_constraint_filters_wrong_type(tmp_settings):
    resolver = _resolver(tmp_settings, _judge(_NO_ONE))

    # "Orange" is the canonical name of a location and an alias of a product.
    assert [m.id for m in resolver.resolve("Orange").matches] == ["l-orange"]
    assert [m.id for m in resolver.resolve("Orange", type="product").matches] == ["p-crush"]






def test_strict_cascade_matches_exact_and_alias_without_llm(tmp_settings):
    resolver = _resolver(tmp_settings, llm=None, cascade=Cascade.STRICT)

    assert resolver.resolve("Texas").matches[0].method == Method.EXACT
    assert resolver.resolve("tx").matches[0].method == Method.ALIAS


def test_strict_cascade_refuses_fuzzy_and_still_suggests(tmp_settings):
    resolver = _resolver(tmp_settings, llm=None, cascade=Cascade.STRICT)

    result = resolver.resolve("Sprite Zeroo")

    assert result.status == Status.NO_MATCH
    assert result.suggestions[0] == "Sprite Zero"




def test_embedding_cache_is_reused_and_rebuilt_when_catalog_changes(tmp_settings, tmp_path):
    cache = tmp_path / "entities.npy"
    llm = _judge(_NO_ONE)

    first = _CountingEmbedder()
    _resolver(tmp_settings, llm, embedder=first, cache_path=cache).resolve("light coke soda")
    assert first.passage_calls == 1
    assert np.load(cache).shape == (8, first.dim)

    # Same catalog: cache hit, passages are not embedded again.
    second = _CountingEmbedder()
    _resolver(tmp_settings, llm, embedder=second, cache_path=cache).resolve("light coke soda")
    assert second.passage_calls == 0

    # New alias changes the catalog hash: recompute.
    changed = _catalog()
    changed[0].aliases.append("Zero Sugar Sprite")
    third = _CountingEmbedder()
    _resolver(tmp_settings, llm, embedder=third, entities=changed, cache_path=cache).resolve("light coke soda")
    assert third.passage_calls == 1


# ------------------------------------------------------------ batched prefetch

@pytest.fixture
def entities() -> list[Entity]:
    return _catalog()


def _batch_llm(handler) -> FakeLLM:
    return FakeLLM({"resolve_judge_batch": handler})


def _blocks(prompt: str) -> dict[str, str]:
    """{"N1": "<mention text>", ...} from a batch prompt."""
    return {
        line.split("]", 1)[0][1:]: line.split("Mention: ", 1)[1]
        for line in prompt.splitlines() if line.startswith("[N")
    }


def test_prefetch_batches_names_and_answers_later_resolves_from_cache(entities, tmp_settings):
    settings = tmp_settings({"resolve": {"judge_batch": 2, "judge_workers": 2}})
    llm = _batch_llm(lambda prompt, tools: "{}")
    resolver = Resolver(entities, FakeEmbedder(), llm, settings, Cascade.FULL)
    names = ["zz one", "zz two", "zz three", "Texas"]  # Texas resolves locally: never sent to the judge

    resolver.prefetch([(name, None) for name in names])
    calls = len(llm.calls)
    results = [resolver.resolve(name) for name in names]

    assert calls == 2  # three unknown names in batches of 2
    assert len(llm.calls) == calls  # resolve() makes no further call
    assert [r.status for r in results] == [Status.NO_MATCH] * 3 + [Status.MATCHED]
    assert all("Texas" not in _blocks(prompt).values() for prompt, _ in llm.calls)


def test_each_name_is_judged_against_its_own_candidates_only(entities, tmp_settings):
    def reply(prompt, tools):
        blocks = _blocks(prompt)
        # Every item answered with a location id: valid only for the location mention.
        return json.dumps({key: {"id": "l-dal", "confidence": 0.95} for key in blocks})

    resolver = Resolver(entities, FakeEmbedder(), _batch_llm(reply), tmp_settings(), Cascade.FULL)
    resolver.prefetch([("fizzy drink thing", "product"), ("big city in dallas county", "location")])

    assert resolver.resolve("fizzy drink thing", "product").status is Status.NO_MATCH
    assert resolver.resolve("big city in dallas county", "location").matches[0].id == "l-dal"


def test_missing_item_in_a_batch_reply_is_no_match(entities, tmp_settings):
    resolver = Resolver(entities, FakeEmbedder(), _batch_llm('{"N2": {"id": "l-dal", "confidence": 0.9}}'),
                        tmp_settings(), Cascade.FULL)
    resolver.prefetch([("big town one", "location"), ("dallas county area", "location")])

    assert resolver.resolve("big town one", "location").status is Status.NO_MATCH
    assert resolver.resolve("dallas county area", "location").matches[0].id == "l-dal"


def test_unusable_batch_is_retried_once_then_listed_as_failures(entities, tmp_settings):
    llm = _batch_llm(["garbage", "still garbage"])
    resolver = Resolver(entities, FakeEmbedder(), llm, tmp_settings(), Cascade.FULL)

    resolver.prefetch([("zz unknown", None)])

    assert len(llm.calls) == 2
    assert resolver.judge_failures == ["zz unknown"]
    assert resolver.resolve("zz unknown").status is Status.NO_MATCH


def test_verdicts_persist_so_a_rebuild_makes_no_batch_call(entities, tmp_settings, tmp_path):
    cache = tmp_path / "entity_embeddings.npy"
    first = _batch_llm('{"N1": {"id": null, "confidence": 0}}')
    Resolver(entities, FakeEmbedder(), first, tmp_settings(), Cascade.FULL, cache).prefetch([("zz one", None)])

    second = _batch_llm("{}")
    Resolver(entities, FakeEmbedder(), second, tmp_settings(), Cascade.FULL, cache).prefetch([("zz one", None)])

    assert len(first.calls) == 1 and second.calls == []




def test_damaged_embedding_cache_is_embedded_again(tmp_settings, tmp_path):
    cache = tmp_path / "entities.npy"
    llm = _judge(_NO_ONE)
    _resolver(tmp_settings, llm, embedder=_CountingEmbedder(), cache_path=cache).resolve("light coke soda")
    cache.write_bytes(cache.read_bytes()[:20])

    again = _CountingEmbedder()
    _resolver(tmp_settings, llm, embedder=again, cache_path=cache).resolve("light coke soda")

    assert again.passage_calls == 1
    assert np.load(cache).shape == (8, again.dim)
