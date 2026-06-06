"""Selection & disclosure tests — hermetic: pure selectors + light embedder, no network.

The selector tests construct :class:`~ir.base.SearchHit`\\ s directly with chosen
scores, so the commit logic is asserted exactly without any retrieval. The
end-to-end :func:`ir.discover` / :func:`ir.eval.evaluate_selection` tests reuse
the disjoint-vocab demo corpus (each query shares tokens with exactly one doc),
so retrieval is unambiguous and the selection numbers are deterministic.
"""

import json
from pathlib import Path

import pytest

import ir
from ir import eval as ev
from ir.base import SearchHit
from ir.select import (
    Disclosure,
    DiscoveryResult,
    Selection,
    abs_threshold,
    disclose,
    discover,
    make_llm_selector,
    rel_threshold,
    score_gap,
    select,
    top_k,
)
from ir.store import CorpusStore

DOCS = {
    "python": "python programming language scripting interpreter",
    "javascript": "javascript browser web frontend dom rendering",
    "database": "sql relational tables rows columns indexing",
    "docker": "docker container image kubernetes orchestration",
}


def _hits(*scores: float) -> list[SearchHit]:
    """SearchHits with the given scores (descending), best-first."""
    return [
        SearchHit(
            artifact_id=f"a{i}",
            surface_kind="capability",
            score=s,
            text=f"text {i}",
            metadata={"name": f"a{i}"},
        )
        for i, s in enumerate(scores)
    ]


def _corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="demo", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


# --------------------------------------------------------------------------- #
# Bare selectors (pure)
# --------------------------------------------------------------------------- #


def test_top_k_selector():
    assert [h.artifact_id for h in top_k(2)(_hits(1.0, 0.9, 0.8))] == ["a0", "a1"]


def test_abs_threshold_selector():
    chosen = abs_threshold(0.5)(_hits(0.9, 0.5, 0.4))
    assert [h.artifact_id for h in chosen] == ["a0", "a1"]


def test_rel_threshold_selector_is_relative_to_top():
    chosen = rel_threshold(0.6)(_hits(1.0, 0.7, 0.5))
    assert [h.artifact_id for h in chosen] == ["a0", "a1"]  # 0.5 < 0.6*1.0 dropped


def test_rel_threshold_guards_nonpositive_top():
    # A non-positive top makes the ratio meaningless -> keep only the best.
    chosen = rel_threshold(0.6)(_hits(-0.1, -0.2))
    assert [h.artifact_id for h in chosen] == ["a0"]
    assert rel_threshold(0.6)([]) == []


def test_score_gap_elbow_cuts_at_cliff():
    # 0.95 stays (>= 0.5*1.0); 0.4 falls off the cliff from 0.95 (< 0.5*0.95).
    chosen = score_gap(0.5)(_hits(1.0, 0.95, 0.4, 0.38))
    assert [h.artifact_id for h in chosen] == ["a0", "a1"]


def test_score_gap_flat_distribution_passes_all():
    # No elbow on a flat list -> nothing is cut (max_k is the over-selection floor).
    chosen = score_gap(0.5)(_hits(0.9, 0.9, 0.9))
    assert len(chosen) == 3


# --------------------------------------------------------------------------- #
# select() — the conservative default + dispatch
# --------------------------------------------------------------------------- #


def test_conservative_keeps_close_hits_and_cuts_on_relative_drop():
    s = select(_hits(1.0, 0.9, 0.5, 0.4))
    assert s.selected_ids == ["a0", "a1"]  # 0.5 < 0.6*1.0 ends the commit
    assert s.reason == "rel_threshold"
    assert s.abstained is False
    assert s.signals["top_score"] == pytest.approx(1.0)
    assert s.signals["n_selected"] == 2
    # min_ratio tracks only *accepted* hits (a1's 0.9), not the rejected cliff (0.5).
    assert s.signals["min_ratio"] == pytest.approx(0.9)


def test_conservative_min_ratio_stays_one_when_only_top_kept():
    # a1 is rejected on the relative threshold, so the accepted set is just a0:
    # min_ratio must stay 1.0 (the rejected 0.5 ratio must not leak in).
    s = select(_hits(1.0, 0.5, 0.4))
    assert s.selected_ids == ["a0"]
    assert s.signals["min_ratio"] == pytest.approx(1.0)


def test_conservative_caps_at_max_k():
    s = select(_hits(*([0.9] * 8)), max_k=3)
    assert len(s.selected) == 3
    assert s.reason == "max_k"


def test_conservative_exhausts_short_close_list():
    s = select(_hits(1.0, 0.95, 0.9))
    assert len(s.selected) == 3
    assert s.reason == "exhausted"


def test_conservative_single_hit():
    s = select(_hits(0.7))
    assert s.selected_ids == ["a0"] and s.reason == "single"


def test_conservative_abstains_on_empty():
    s = select(_hits())
    assert s.abstained and s.selected == [] and s.reason == "abstain:no_candidates"


def test_conservative_abstains_below_floor():
    s = select(_hits(0.2, 0.1), min_score=0.5)
    assert s.abstained and s.reason == "abstain:below_floor"
    assert s.signals["min_score"] == pytest.approx(0.5)


def test_conservative_nonpositive_top_keeps_single_best():
    s = select(_hits(-0.1, -0.2, -0.05))
    assert s.selected_ids == ["a0"] and s.reason == "nonpositive_top"


def test_select_dispatch_named_and_callable():
    # named score_gap strategy threads gap_ratio through
    s = select(_hits(1.0, 0.95, 0.4), strategy="score_gap", gap_ratio=0.5)
    assert s.selected_ids == ["a0", "a1"]
    # a bare callable Selector is accepted and labelled "custom"
    s2 = select(_hits(1.0, 0.9, 0.8), strategy=top_k(1))
    assert s2.selected_ids == ["a0"] and s2.reason == "custom"


def test_select_unknown_strategy_raises():
    with pytest.raises(ValueError):
        select(_hits(1.0), strategy="bogus")


def test_select_abs_threshold_requires_min_score():
    with pytest.raises(ValueError):
        select(_hits(1.0), strategy="abs_threshold")
    s = select(_hits(0.9, 0.4), strategy="abs_threshold", min_score=0.5)
    assert s.selected_ids == ["a0"]


def test_selection_to_dict_is_json_serializable():
    s = select(_hits(1.0, 0.9, 0.5))
    blob = json.dumps(s.to_dict())
    back = json.loads(blob)
    assert back["selected_ids"] == ["a0", "a1"]
    assert back["abstained"] is False
    assert isinstance(back["selected"][0]["score"], float)


# --------------------------------------------------------------------------- #
# LLM-as-selector — injectable, offline, with fallback
# --------------------------------------------------------------------------- #


def test_llm_selector_uses_injected_chooser():
    sel = make_llm_selector("find a", chooser=lambda *, query, candidates: ["a1"])
    assert [h.artifact_id for h in sel(_hits(0.9, 0.8, 0.7))] == ["a1"]


def test_llm_selector_falls_back_on_empty_or_error():
    # empty pick -> conservative fallback
    empty = make_llm_selector("x", chooser=lambda **k: [])
    assert [h.artifact_id for h in empty(_hits(1.0, 0.9, 0.4))] == ["a0", "a1"]
    # raising chooser -> conservative fallback (never propagates)
    boom = make_llm_selector(
        "x", chooser=lambda **k: (_ for _ in ()).throw(RuntimeError)
    )
    assert [h.artifact_id for h in boom(_hits(1.0, 0.9, 0.4))] == ["a0", "a1"]


def test_llm_selector_as_select_strategy():
    sel = make_llm_selector("q", chooser=lambda **k: ["a2"])
    s = select(_hits(0.9, 0.8, 0.7), strategy=sel)
    assert s.selected_ids == ["a2"]


# --------------------------------------------------------------------------- #
# Progressive disclosure
# --------------------------------------------------------------------------- #


def _skill_hit(skill_path: str) -> SearchHit:
    return SearchHit(
        "sk",
        "capability",
        0.9,
        "sk: short description",
        {"name": "sk", "skill_path": skill_path},
    )


def test_disclose_metadata_level_loads_no_body(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("# Body\nfull skill body")
    d = disclose(select([_skill_hit(str(p))]), level="metadata")[0]
    assert d.body is None
    assert d.pointer == str(p)
    assert d.summary == "sk: short description"
    assert d.name == "sk"


def test_disclose_body_level_loads_pointer(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("# Body\nfull skill body")
    d = disclose(select([_skill_hit(str(p))]), level="body")[0]
    assert "full skill body" in d.body
    assert d.summary == "sk: short description"  # cheap summary still present


def test_disclose_directory_pointer_reads_readme(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "README.md").write_text("package readme contents")
    hit = SearchHit(
        "pkg", "description", 0.8, "pkg: desc", {"name": "pkg", "path": str(pkg)}
    )
    d = disclose(select([hit]), level="body")[0]
    assert d.body == "package readme contents"


def test_disclose_tolerates_stale_pointer():
    hit = _skill_hit("/no/such/file/SKILL.md")
    d = disclose(select([hit]), level="body")[0]
    assert d.body is None
    assert d.metadata.get("disclosure") == "pointer_unreadable"


def test_disclose_unknown_level_raises():
    with pytest.raises(ValueError):
        disclose(select(_hits(1.0)), level="bogus")


def test_disclose_is_pure_does_not_mutate_selection(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("body")
    sel = select([_skill_hit(str(p))])
    before = sel.selected[0].metadata.copy()
    disclose(sel, level="body")
    assert sel.selected[0].metadata == before  # original hit untouched


def test_disclose_custom_loader():
    hit = SearchHit("x", "capability", 0.5, "summary", {"name": "x"})
    out = disclose(select([hit]), level="body", loader=lambda meta: "INJECTED")[0]
    assert out.body == "INJECTED"


def test_disclosure_to_dict_json_serializable(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("body text")
    d = disclose(select([_skill_hit(str(p))]), level="body")[0]
    assert json.loads(json.dumps(d.to_dict()))["body"] == "body text"


# --------------------------------------------------------------------------- #
# discover() — the single search-and-select tool
# --------------------------------------------------------------------------- #


def test_discover_end_to_end_commits_to_one():
    res = discover(_corpus(), "python programming language", k=4, mode="dense")
    assert isinstance(res, DiscoveryResult)
    assert res.ids == ["python"]
    assert res.abstained is False
    assert res.n_retrieved >= 1


def test_discover_result_is_json_serializable():
    res = discover(_corpus(), "docker container image", k=4, mode="dense")
    blob = json.loads(json.dumps(res.to_dict()))
    assert blob["results"][0]["artifact_id"] == "docker"
    assert blob["n_selected"] == len(res.results)
    assert isinstance(blob["signals"], dict)


def test_discover_abstains_with_floor():
    res = discover(_corpus(), "python programming", k=4, mode="dense", min_score=2.0)
    assert res.abstained and res.results == [] and res.reason == "abstain:below_floor"


def test_discover_accepts_corpus_name(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("deploy the app to the server with systemd units")
    from ir import cli

    cli.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    cli.build("notes", embedder="light")
    res = discover("notes", "deploy app systemd", k=3, mode="dense")
    assert "deploy.md" in res.ids


# --------------------------------------------------------------------------- #
# evaluate_selection — conditional commit rate, isolated from retrieval
# --------------------------------------------------------------------------- #


def test_evaluate_selection_reports_perfect_on_disjoint_corpus():
    cases = [
        ev.DiscoveryCase("python programming language", gold=("python",)),
        ev.DiscoveryCase("docker container image kubernetes", gold=("docker",)),
    ]
    rep = ev.evaluate_selection(_corpus(), cases, mode="dense", k=4)
    assert rep.n_gold == 2 and rep.n_abstention == 0
    assert rep.conditional_commit_rate == pytest.approx(1.0)
    assert rep.n_gold_retrieved == 2
    assert rep.selection_recall == pytest.approx(1.0)
    assert rep.selection_precision == pytest.approx(1.0)
    assert rep.mean_selected_size == pytest.approx(1.0)
    assert "SelectionReport" in str(rep)
    assert json.loads(json.dumps(rep.to_dict()))["k"] == 4


def test_conditional_commit_rate_isolates_selection_from_retrieval():
    # A query whose gold retrieval cannot surface (gold token absent from query)
    # must NOT count against the conditional rate — it is a retrieval miss.
    cases = [
        ev.DiscoveryCase(
            "python programming language", gold=("python",)
        ),  # retrievable
        ev.DiscoveryCase(
            "xyzzy nonsense tokens", gold=("docker",)
        ),  # gold not surfaced
    ]
    rep = ev.evaluate_selection(_corpus(), cases, mode="dense", k=2)
    # only the first case has gold among the retrieved candidates
    assert rep.n_gold_retrieved == 1
    assert rep.conditional_commit_rate == pytest.approx(1.0)
    # recall/precision/F1 share that denominator — the un-retrieved case is NOT
    # averaged in as a 0.0 (which would contaminate selection with retrieval).
    assert rep.selection_recall == pytest.approx(1.0)
    assert rep.selection_precision == pytest.approx(1.0)
    assert rep.selection_f1 == pytest.approx(1.0)


def test_evaluate_selection_abstention_accuracy_with_floor():
    cases = [
        ev.DiscoveryCase("python programming language", gold=("python",)),
        ev.DiscoveryCase("zzz unrelated nonsense query", gold=()),  # abstention
    ]
    # No floor: the conservative selector never abstains -> abstention accuracy 0.
    no_floor = ev.evaluate_selection(_corpus(), cases, mode="dense", k=4)
    assert no_floor.n_abstention == 1
    assert no_floor.abstention_accuracy == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# CLI — discover + eval-select (hermetic, light embedder)
# --------------------------------------------------------------------------- #


def test_cli_discover_and_eval_select(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    from ir import cli

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("deploy the app to the server with systemd units")
    (docs / "baking.md").write_text("bake a cake in the oven with flour and sugar")
    cli.register("notes", "files", root=str(docs), pattern=r".*\.md$")
    cli.build("notes", embedder="light")

    out = cli.discover("notes", "deploy app systemd units", k=3, mode="dense")
    assert "deploy.md" in out

    # --disclose loads the body via the stored path pointer
    out_disclosed = cli.discover(
        "notes", "deploy app systemd units", k=3, mode="dense", disclose=True
    )
    assert "systemd" in out_disclosed

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"query": "deploy app systemd", "gold": ["deploy.md"]}) + "\n"
    )
    sel_out = cli.eval_select("notes", str(cases), mode="dense", k=3)
    assert "SelectionReport" in sel_out and "conditional commit rate" in sel_out

    # empty corpus is guided, not crashed
    blank_dir = tmp_path / "blankdir"
    blank_dir.mkdir()
    cli.register("blank", "files", root=str(blank_dir), pattern=r".*\.md$")
    cli.build("blank", embedder="light")
    assert "build it first" in cli.discover("blank", "anything")
    assert "build it first" in cli.eval_select("blank", str(cases))


# --------------------------------------------------------------------------- #
# Public surface reachable from the package root
# --------------------------------------------------------------------------- #


def test_public_exports():
    for name in (
        "select",
        "disclose",
        "discover",
        "Selection",
        "Disclosure",
        "DiscoveryResult",
    ):
        assert hasattr(ir, name), name
    assert ir.Selection is Selection
