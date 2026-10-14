"""Tests for ``CorpusSource.from_mcp_tools`` (#74): a tools corpus factory.

Each tool is indexed on ``name + description`` only; its parameter schema never
enters the embedded text, but it does enter the SHA-256 invalidation key so a
schema change re-embeds the tool. Input is duck-typed (mappings or objects), so
``ir`` takes no MCP dependency. Hermetic: light embedder, in-memory store.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

import ir
from ir.base import ledger_key
from ir.sources import mcp_tool_signal
from ir.store import CorpusStore

TOOLS = [
    {
        "name": "fetch_url",
        "description": "Fetch a web page over HTTP and return its text",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "ZQXSCHEMA"}},
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        },
    },
    {"name": "list_files", "description": "List the files in a directory"},
]


def _build(tools, store=None):
    src = ir.CorpusSource.from_mcp_tools(tools)
    store = store if store is not None else CorpusStore.memory()
    return ir.build(src, store=store, embedder="light"), store


def test_indexes_name_and_description_only():
    corpus, _ = _build(TOOLS)
    assert len(corpus) == 3  # one capability surface per tool
    hit = ir.search(corpus, "fetch a web page over http", k=1)[0]
    assert hit.artifact_id == "fetch_url"
    assert hit.text == "fetch_url\n\nFetch a web page over HTTP and return its text"
    # The parameter schema stays out of every embedded text ...
    every = ir.search(corpus, "anything", k=10, per_artifact=False)
    assert all("ZQXSCHEMA" not in h.text and "properties" not in h.text for h in every)
    # ... and is not lexically searchable either.
    assert all(
        "ZQXSCHEMA" not in h.text
        for h in ir.search(corpus, "ZQXSCHEMA", k=10, mode="lexical")
    )


def test_hits_carry_name_and_param_names_metadata():
    corpus, _ = _build(TOOLS)
    by_id = {h.artifact_id: h for h in ir.search(corpus, "do something", k=10)}
    assert by_id["fetch_url"].metadata["param_names"] == ["url"]
    assert by_id["send_email"].metadata["param_names"] == ["to", "body"]
    assert by_id["list_files"].metadata["param_names"] == []
    # ``name`` is a hard filter field (exact match), as for skills.
    hits = ir.search(corpus, "do something", k=10, filter={"name": "send_email"})
    assert [h.artifact_id for h in hits] == ["send_email"]


def test_signal_is_sha256_over_name_description_parameters():
    tool = TOOLS[0]
    canonical = json.dumps(
        [tool["name"], tool["description"], tool["parameters"]],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert mcp_tool_signal("fetch_url", tool) == expected
    # Key order inside the schema does not matter (canonical JSON).
    reordered = dict(tool, parameters=dict(reversed(list(tool["parameters"].items()))))
    assert mcp_tool_signal("fetch_url", reordered) == expected


def test_schema_change_reembeds_but_unrelated_fields_do_not():
    store = CorpusStore.memory()
    _build(TOOLS, store)
    v1 = store.get_ledger_entry(ledger_key("send_email"))["version"]
    other = store.get_ledger_entry(ledger_key("fetch_url"))["version"]

    # A field outside (name, description, parameters) does not invalidate.
    decorated = [dict(t, annotations={"readOnlyHint": True}) for t in TOOLS]
    _build(decorated, store)
    assert store.get_ledger_entry(ledger_key("send_email"))["version"] == v1

    # A parameter-schema change does, even though the embedded text is unchanged.
    changed = [dict(t) for t in TOOLS]
    changed[1]["parameters"] = {
        "type": "object",
        "properties": {"to": {"type": "string"}, "cc": {"type": "string"}},
    }
    _build(changed, store)
    assert store.get_ledger_entry(ledger_key("send_email"))["version"] != v1
    assert store.get_ledger_entry(ledger_key("fetch_url"))["version"] == other


def test_duck_typed_inputs():
    # Objects with attributes (fastmcp-like ``parameters``), the MCP wire field
    # ``inputSchema``, the Anthropic API's ``input_schema``, and a {name: tool}
    # mapping all normalize to the same records.
    fastmcp_like = SimpleNamespace(
        name="fetch_url",
        description=TOOLS[0]["description"],
        parameters=TOOLS[0]["parameters"],
    )
    wire = {
        "name": "fetch_url",
        "description": TOOLS[0]["description"],
        "inputSchema": TOOLS[0]["parameters"],
    }
    api = {
        "name": "fetch_url",
        "description": TOOLS[0]["description"],
        "input_schema": TOOLS[0]["parameters"],
    }
    expected = mcp_tool_signal("fetch_url", TOOLS[0])
    for tool in (fastmcp_like, wire, api):
        src = ir.CorpusSource.from_mcp_tools([tool])
        assert src.change_signal("fetch_url", src.scope["fetch_url"]) == expected
    by_name = ir.CorpusSource.from_mcp_tools({t["name"]: t for t in TOOLS})
    assert sorted(by_name.scope) == sorted(t["name"] for t in TOOLS)


def test_fetcher_reference_and_missing_description():
    src = ir.CorpusSource.from_mcp_tools(lambda: [{"name": "noop"}], name="mine")
    assert src.name == "mine"
    assert src.scope["noop"] == {"name": "noop", "description": "", "parameters": {}}


def test_bad_inputs_fail_loudly():
    with pytest.raises(ValueError, match="no name"):
        ir.CorpusSource.from_mcp_tools([{"description": "anonymous"}])
    with pytest.raises(ValueError, match="duplicate tool name 'a'"):
        ir.CorpusSource.from_mcp_tools([{"name": "a"}, {"name": "a"}])


def test_mapping_keys_name_records_without_a_name():
    src = ir.CorpusSource.from_mcp_tools(
        {"ping": {"description": "Check the server is up"}, "alias": {"name": "pong"}}
    )
    assert sorted(src.scope) == ["ping", "pong"]  # a record's own name wins
    assert src.scope["ping"]["description"] == "Check the server is up"


def test_metadata_of_can_be_overridden():
    src = ir.CorpusSource.from_mcp_tools(
        TOOLS, metadata_of=lambda aid, raw: {"server": "s1"}
    )
    assert src.metadata_of("fetch_url", src.scope["fetch_url"]) == {"server": "s1"}
