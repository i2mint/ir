"""Sibling cases of recent features that their own tests did not cover.

Each test pins one case one step away from the example its feature was built for:
newline-separated words, a blank chunk from a custom splitter, a non-default
fusion's calibrated floor, a synopsis under a folding normalizer, and a grep corpus
built from a relative root.
"""

import os
import warnings

import pytest

import ir
from ir.store import CorpusStore
from ir.strategy import Chunked, Fields, balanced_windows, fold_text, word_windows


def test_word_windows_cut_at_any_whitespace():
    text = "alpha\nbeta\ngamma\ndelta\nepsilon\tzeta"
    windows = word_windows(text, size=12)
    assert windows == ["alpha\nbeta", "gamma\ndelta", "epsilon\tzeta"]
    words = set(text.split())
    assert all(set(w.split()) <= words for w in windows)  # no word cut in two


def test_word_windows_overlap_starts_at_a_word_after_a_newline():
    windows = word_windows("aaaa\nbbbb\ncccc\ndddd", size=9, overlap=4)
    assert all(not w[0].isspace() for w in windows)
    assert {t for w in windows for t in w.split()} == {"aaaa", "bbbb", "cccc", "dddd"}


def test_chunked_with_a_splitter_yields_no_blank_surfaces():
    text = "abc" + " " * 30 + "def"
    plan = Chunked(chunk_size=10, overlap=2, split=balanced_windows).decompose("x", text)
    texts = [s.text for s in plan.surfaces]
    assert texts and all(t.strip() for t in texts)
    assert [s.metadata["chunk_index"] for s in plan.surfaces] == list(range(len(texts)))
    assert {s.metadata["n_chunks"] for s in plan.surfaces} == {len(texts)}


def _calibration_corpus(name):
    from ir.eval import DiscoveryCase

    docs = {
        "a": "alpha apple pie",
        "b": "banana bread loaf",
        "c": "cherry tart dessert",
        "d": "durian fruit smell",
    }
    corpus = ir.build(
        ir.CorpusSource.from_mapping(docs, name=name, strategy=ir.WholeText()),
        store=CorpusStore.memory(),
        embedder="light",
    )
    cases = [
        DiscoveryCase("alpha apple", ("a",)),
        DiscoveryCase("banana bread", ("b",)),
        DiscoveryCase("cherry tart", ("c",)),
        DiscoveryCase("quantum chromodynamics", ()),
        DiscoveryCase("tax law filing", ()),
    ]
    return corpus, cases


def test_a_blend_floor_has_its_own_key():
    from ir.eval import calibrate_min_score, load_calibration
    from ir.select import _resolve_auto_min_score

    corpus, cases = _calibration_corpus("cal_fusion")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rrf = calibrate_min_score(corpus, cases, mode="hybrid", persist=True)
        # No blend floor yet: the rrf floor must not be applied to blend scores.
        assert _resolve_auto_min_score(corpus, "hybrid", fusion="blend") is None
        blend = calibrate_min_score(
            corpus, cases, mode="hybrid", persist=True, fusion="blend"
        )
    assert blend.fusion == "blend"
    # Calibrating under blend does not overwrite the rrf floor.
    assert corpus.store.get_calibration("hybrid")["min_score"] == rrf.min_score
    assert corpus.store.get_calibration("hybrid~blend")["min_score"] == blend.min_score
    assert load_calibration(corpus, "hybrid", fusion="blend").min_score == blend.min_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _resolve_auto_min_score(corpus, "hybrid") == rrf.min_score
        assert (
            _resolve_auto_min_score(corpus, "hybrid", fusion="blend")
            == blend.min_score
        )


def test_synopsis_is_folded_like_the_query():
    docs = {"d1": {"about": "a baked custard"}, "d2": {"about": "a noodle soup"}}
    synopses = {"d1": "Crème Brûlée, the French dessert", "d2": "Phở, Vietnamese soup"}
    strategy = ir.with_synopsis(
        Fields(["about"], normalize=fold_text),
        synthesize=lambda a: synopses[a.id],
        synthesizer_id="test",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corpus = ir.build(
            ir.CorpusSource.from_mapping(docs, name="syn_fold", strategy=strategy),
            store=CorpusStore.memory(),
            embedder="light",
        )
        for query, aid in [("Crème", "d1"), ("Phở", "d2")]:
            hits = ir.search(corpus, query, mode="lexical")
            assert (aid, "synopsis") in [(h.artifact_id, h.surface_kind) for h in hits]


def test_grep_corpus_from_a_relative_root_works_from_any_cwd(tmp_path, monkeypatch):
    (tmp_path / "notes" / "sub").mkdir(parents=True)
    (tmp_path / "notes" / "a.md").write_text("# Acme\nacme\n")
    (tmp_path / "notes" / "sub" / "b.md").write_text("acme here\n")
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corpus = ir.build(
            ir.CorpusSource.from_files("notes", name="rel_grep", pattern=r"\.md$"),
            store=CorpusStore.memory(),
            embedder="light",
        )
    stored = {
        m["artifact_id"]: m["metadata"]["path"] for m in corpus.store.metas()[1]
    }
    assert all(os.path.isabs(p) for p in stored.values())
    monkeypatch.chdir(tmp_path.anchor)
    hits = ir.search(corpus, "acme", mode="grep")
    # Artifact ids use the OS separator (dol's file keys): compare them as posix.
    assert sorted(h.artifact_id.replace(os.sep, "/") for h in hits) == [
        "a.md",
        "sub/b.md",
    ]


def test_grep_filter_sees_the_stored_metadata(tmp_path):
    (tmp_path / "a.md").write_text("acme\n")
    (tmp_path / "b.md").write_text("acme too\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corpus = ir.build(
            ir.CorpusSource.from_files(tmp_path, name="filt_grep", pattern=r"\.md$"),
            store=CorpusStore.memory(),
            embedder="light",
        )
    stored = {
        m["artifact_id"]: m["metadata"]["path"] for m in corpus.store.metas()[1]
    }
    for mode in ("lexical", "grep"):
        hits = ir.search(corpus, "acme", mode=mode, filter={"path": stored["b.md"]})
        assert [h.artifact_id for h in hits] == ["b.md"], mode
