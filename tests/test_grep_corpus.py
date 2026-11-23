"""Corpus-bound ``mode="grep"`` and per-line (field) scores (#152)."""

import json
import re

import pytest

import ir
from ir.grep import corpus_paths, grep_hits, json_field_score, markdown_heading_score
from ir.store import CorpusStore


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "acme.md").write_text("# Acme invoices\nnothing here\n")
    (root / "mention.md").write_text("we paid acme\nand acme again\n")
    (root / "other.md").write_text("# Unrelated\nno match\n")
    (root / "skip.txt").write_text("acme in a file the corpus did not index\n")
    return root


@pytest.fixture
def corpus(tree):
    source = ir.CorpusSource.from_files(tree, name="notes", pattern=r"\.md$")
    return ir.build(source, store=CorpusStore.memory(), embedder="light")


def test_corpus_paths_maps_the_indexed_artifacts_to_their_files(corpus, tree):
    assert corpus_paths(corpus) == {
        name: tree / name for name in ("acme.md", "mention.md", "other.md")
    }


def test_grep_mode_reads_the_corpus_files_under_its_artifact_ids(corpus):
    hits = ir.search(corpus, "acme", mode="grep")
    assert [(h.artifact_id, h.score) for h in hits] == [
        ("mention.md", 2.0),
        ("acme.md", 1.0),
    ]
    assert hits[0].surface_kind == "grep" and hits[0].source == "notes"
    assert hits[0].metadata["path"] == "mention.md"  # no absolute local path


def test_grep_mode_reads_files_live_not_the_index(corpus, tree):
    (tree / "other.md").write_text("acme\nacme\nacme\n")
    assert ir.search(corpus, "acme", mode="grep", k=1)[0].artifact_id == "other.md"


def test_grep_mode_filters_on_the_stored_metadata(corpus):
    hits = ir.search(corpus, "acme", mode="grep", filter={"filename": "acme.md"})
    assert [h.artifact_id for h in hits] == ["acme.md"]


def test_grep_mode_takes_grep_options_and_refuses_index_only_ones(corpus):
    score = markdown_heading_score(5.0)
    hits = ir.search(corpus, "acme", mode="grep", grep={"score_line": score})
    assert [(h.artifact_id, h.score) for h in hits] == [
        ("acme.md", 5.0),
        ("mention.md", 2.0),
    ]
    with pytest.raises(ValueError, match="rerank"):
        ir.search(corpus, "acme", mode="grep", rerank=lambda q, s: [1.0] * len(s))
    with pytest.raises(ValueError, match="paths"):
        ir.search(corpus, "acme", mode="grep", grep={"paths": {}})
    with pytest.raises(ValueError, match="mode='grep' only"):
        ir.search(corpus, "acme", mode="lexical", grep={"fixed": True})


def test_grep_mode_needs_a_file_backed_corpus():
    src = ir.CorpusSource.from_mapping({"a": {"text": "acme"}}, name="mem")
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    with pytest.raises(ValueError, match="file-backed"):
        ir.search(corpus, "acme", mode="grep")


def test_paths_and_roots_are_exclusive(tree):
    with pytest.raises(ValueError, match="exactly one"):
        grep_hits("acme", roots=tree, paths={"a": tree / "acme.md"})
    with pytest.raises(ValueError, match="exactly one"):
        grep_hits("acme")
    with pytest.raises(ValueError, match="id_of"):
        grep_hits("acme", paths={"a": tree / "acme.md"}, id_of=lambda r, p: "x")


def test_paths_skip_missing_files_and_honour_glob(tree):
    paths = {"a": tree / "acme.md", "gone": tree / "gone.md", "t": tree / "skip.txt"}
    assert [h.artifact_id for h in grep_hits("acme", paths=paths)] == ["a", "t"]
    assert [h.artifact_id for h in grep_hits("acme", paths=paths, glob="a")] == ["a"]


def test_json_field_score_puts_a_title_match_above_a_body_match(tmp_path):
    def jsonl(name, *records):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    paths = {
        "titled": jsonl("t.jsonl", {"title": "Acme", "body": "invoice"}),
        "mentioned": jsonl(
            "m.jsonl",
            {"title": "Invoice", "body": "acme"},
            {"title": "Receipt", "body": "acme"},
        ),
    }
    plain = grep_hits("acme", paths=paths)
    assert [h.artifact_id for h in plain] == ["mentioned", "titled"]  # 2 lines vs 1
    weighted = grep_hits(
        "acme", paths=paths, score_line=json_field_score({"title": 3.0, "body": 1.0})
    )
    assert [(h.artifact_id, h.score) for h in weighted] == [
        ("titled", 3.0),
        ("mentioned", 2.0),
    ]


def test_a_bad_line_score_is_an_error(tree):
    with pytest.raises(ValueError, match="score_line"):
        grep_hits("acme", roots=tree, score_line=lambda line, rx: -1)


def test_json_field_score_ignores_keys_and_non_strings():
    score = json_field_score({"acme": 9.0})
    rx = re.compile("acme")
    assert score('{"acme": 3}', rx) == 0.0
    assert score("acme, not json", rx) == 1.0
    assert score('["acme"]', rx) == 1.0
