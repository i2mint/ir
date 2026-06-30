"""Tests for the agent-callable tool surface (``ir.tools``)."""

import json

import ir
from ir.store import CorpusStore


def _corpus():
    docs = {
        "deploy": "deploy the app to the server with systemd units",
        "embed": "embed text with a model and cache the vectors",
        "search": "vector similarity search with metadata filters",
    }
    return ir.build(
        ir.CorpusSource.from_mapping(docs, name="t_tools", strategy=ir.WholeText()),
        store=CorpusStore.memory(),
        embedder="light",
    )


def test_search_returns_jsonable_dict():
    out = ir.tools.search("deploy the app to the server", corpus=_corpus(), k=2)
    assert isinstance(out, dict)
    json.dumps(out)  # must be JSON-serializable (MCP/HTTP-ready)


def test_make_search_binds_corpus_and_names_the_tool():
    fn = ir.tools.make_search(_corpus(), name="demo-corpus")
    assert fn.__name__ == "search_demo_corpus"  # sanitized
    assert "demo-corpus" in (fn.__doc__ or "")
    out = fn("deploy the app")
    assert isinstance(out, dict)
    json.dumps(out)


def test_top_level_aliases_exist():
    assert ir.search_tool is ir.tools.search
    assert ir.make_search is ir.tools.make_search
