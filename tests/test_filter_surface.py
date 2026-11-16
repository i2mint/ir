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


def _ids(result):
    return sorted(r["id"] if "id" in r else r["name"] for r in result["results"])


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
