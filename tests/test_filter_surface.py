"""Metadata filters from the CLI (``--filter`` JSON) and ``make_search`` (#138)."""

import pytest

import ir
from ir import cli, tools
from ir.store import CorpusStore

DOCS = {"a": "alpha apple", "b": "alpha banana", "c": "alpha cherry"}
META = {"a": {"project": "p1"}, "b": {"project": "p2"}, "c": {"project": "p1"}}


@pytest.fixture
def corpus():
    src = ir.CorpusSource.from_mapping(
        DOCS, name="filt", strategy=ir.WholeText(), metadata_of=lambda i, raw: META[i]
    )
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_filter_kw_parses_json_and_is_empty_when_unset():
    assert cli._filter_kw(None) == {}
    assert cli._filter_kw("") == {}
    assert cli._filter_kw('{"project": "p1"}') == {"filter": {"project": "p1"}}
    assert cli._filter_kw({"a": 1}) == {"filter": {"a": 1}}


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"x"'])
def test_filter_kw_rejects_non_object(bad):
    with pytest.raises(SystemExit, match="--filter"):
        cli._filter_kw(bad)


def test_cli_search_and_discover_apply_filter(monkeypatch, corpus):
    monkeypatch.setattr(cli, "_open_ready", lambda name: corpus)
    out = cli.search("filt", "alpha", mode="lexical", filter='{"project": "p2"}')
    assert out.count("\n") == 0 and "  b  " in out
    unfiltered = cli.search("filt", "alpha", mode="lexical")
    assert unfiltered.count("\n") == 2
    out = cli.discover(
        "filt", "alpha", mode="lexical", strategy="top_k", filter='{"project": "p2"}'
    )
    assert "selected 1/1" in out


def test_make_search_bound_filter(corpus):
    f = tools.make_search(corpus, mode="lexical", filter={"project": "p1"})
    res = f("alpha")
    assert len(res["results"]) == 2
    assert "filter" not in f.__code__.co_varnames[: f.__code__.co_argcount]


def test_make_search_exposed_filter_is_anded_with_bound(corpus):
    f = tools.make_search(
        corpus, mode="lexical", filter={"project": "p1"}, expose_filter=True
    )
    assert len(f("alpha")["results"]) == 2
    assert len(f("alpha", filter={"project": "p2"})["results"]) == 0
    assert len(f("alpha", filter={"project": {"$ne": "p2"}})["results"]) == 2


def test_short_f_still_means_fusion_and_filter_is_long_only():
    from ir.__main__ import mk_parser

    ns = mk_parser().parse_args(["search", "n", "q", "-f", "blend", "--filter", "{}"])
    assert (ns.fusion, ns.filter) == ("blend", "{}")


@pytest.mark.parametrize(
    "bad",
    ['{"project": {"$like": "p%"}}', '{"$and": {"project": "p1"}}'],
)
def test_filter_kw_rejects_unsupported_operators(bad):
    with pytest.raises(SystemExit, match="--filter"):
        cli._filter_kw(bad)


def test_filter_kw_empty_object_means_no_filter():
    assert cli._filter_kw("{}") == {}
    assert cli._filter_kw("  ") == {}


def test_python_api_raises_on_unsupported_operator(corpus):
    from vd.base import UnsupportedFilterError

    with pytest.raises(UnsupportedFilterError):
        corpus.search("alpha", mode="lexical", filter={"project": {"$like": "p%"}})


def test_discover_short_f_is_fusion():
    from ir.__main__ import mk_parser

    ns = mk_parser().parse_args(["discover", "n", "q", "-f", "blend"])
    assert ns.fusion == "blend"


# ----- --variant-weight (#148) --------------------------------------------- #


def test_variant_weight_needs_expand():
    with pytest.raises(SystemExit, match="--expand"):
        cli._expand_kw(False, "c", 0.5)
    assert cli._expand_kw(False, "c") == {}
    assert cli._expand_kw(False, "c", 1.0) == {}


@pytest.mark.parametrize("bad", [2, -0.5, float("nan"), float("inf")])
def test_variant_weight_out_of_range_is_a_clean_exit(bad):
    with pytest.raises(SystemExit, match=r"\[0, 1\]"):
        cli._expand_kw(True, "c", bad)


def test_variant_weight_is_forwarded_with_expand(monkeypatch):
    kw = cli._expand_kw(True, "c", 0.5)
    assert kw["variant_weight"] == 0.5 and callable(kw["formulate"])
    assert "variant_weight" not in cli._expand_kw(True, "c")  # default: today's merge


def test_cli_search_and_discover_parse_and_apply_variant_weight(
    monkeypatch, tmp_path, corpus
):
    from ir.__main__ import mk_parser

    ns = mk_parser().parse_args(["search", "n", "q", "-e", "--variant-weight", "0.5"])
    assert (ns.expand, ns.variant_weight) == (True, 0.5)
    ns = mk_parser().parse_args(
        ["discover", "n", "q", "-e", "--variant-weight", "0.25"]
    )
    assert ns.variant_weight == 0.25

    ledger = tmp_path / "s.jsonl"
    ledger.write_text('{"terms": ["banana", "plantain"]}\n')
    monkeypatch.setenv("IR_SYNONYMS", str(ledger))
    monkeypatch.setattr(cli, "_open_ready", lambda name: corpus)
    base = cli.search("filt", "plantain", mode="lexical", expand=True)
    weighted = cli.search(
        "filt", "plantain", mode="lexical", expand=True, variant_weight=0.5
    )
    assert "  b  " in base and "  b  " in weighted  # the variant still finds b
    s_base = float(base.split()[0])
    s_weighted = float(weighted.split()[0])
    assert s_weighted == pytest.approx(s_base * 0.5, rel=1e-2)


def test_variant_weight_demotes_only_the_variants_end_to_end(monkeypatch, tmp_path):
    docs = {"b": "banana bread", "d": "plantain stew", "c": "cherry pie"}
    src = ir.CorpusSource.from_mapping(docs, name="vw", strategy=ir.WholeText())
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    ledger = tmp_path / "s.jsonl"
    ledger.write_text('{"terms": ["banana", "plantain"]}\n')
    monkeypatch.setenv("IR_SYNONYMS", str(ledger))
    monkeypatch.setattr(cli, "_open_ready", lambda name: corpus)

    def scores(**kw):
        out = cli.search("vw", "plantain", mode="lexical", expand=True, **kw)
        return {ln.split()[1]: float(ln.split()[0]) for ln in out.splitlines()}

    base, weighted = scores(), scores(variant_weight=0.5)
    assert weighted["d"] == pytest.approx(base["d"])  # the query's own hit is untouched
    assert weighted["b"] == pytest.approx(
        base["b"] * 0.5, rel=1e-2
    )  # the variant's halves
