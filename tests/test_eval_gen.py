"""Case-generation tests — hermetic: deterministic stub generators, no LLM/network.

The LLM is injected (`query_generator` / `abstention_generator`), so masking, the
leakage guard, gold assignment, the abstention fraction, and end-to-end
scorability are all tested without `oa` or a model.
"""

import warnings

import pytest

import ir
from ir import eval as ev
from ir import eval_gen as eg
from ir.store import CorpusStore


def _echo_gen(description, *, n):
    """Stub query generator: echo the (masked) description as n identical intents."""
    return [description] * n


def _skill_source(docs):
    return ir.CorpusSource.from_mapping(docs, name="g", strategy=ir.Skill())


# --------------------------------------------------------------------------- #
# Name masking / leakage detection
# --------------------------------------------------------------------------- #


def test_mask_name_hyphen_and_underscore_variants():
    assert (
        eg.mask_name("Use my-packages now", "my-packages") == "Use this capability now"
    )
    assert eg.mask_name("run my_tool please", "my_tool") == "run this capability please"
    # case-insensitive, and the de-hyphenated form matches "CI setup"
    assert eg.mask_name("CI setup helps", "ci-setup") == "this capability helps"


def test_mask_name_does_not_overmatch_substrings():
    # a short name must not blast through unrelated words
    assert (
        eg.mask_name("specific scientific terms", "ci") == "specific scientific terms"
    )


def test_leaks_name_detection():
    assert eg._leaks_name("please use projreg now", "projreg")
    assert not eg._leaks_name("please use the registry now", "projreg")


def test_parse_lines_strips_bullets_and_numbering():
    text = '1. first\n- second\n  * third \n\n"fourth"'
    assert eg._parse_lines(text) == ["first", "second", "third", "fourth"]


# --------------------------------------------------------------------------- #
# generate_cases
# --------------------------------------------------------------------------- #


def test_generate_cases_basic_shape():
    docs = {
        "alpha": {
            "name": "alpha",
            "description": "alpha tool for foo widget tasks here",
        },
        "beta": {"name": "beta", "description": "beta tool for bar gadget tasks here"},
    }
    cases = eg.generate_cases(
        _skill_source(docs), k=2, query_generator=_echo_gen, corpus_name="g"
    )
    assert len(cases) == 4  # 2 per artifact
    assert {c.gold[0] for c in cases} == {"alpha", "beta"}
    assert all(len(c.gold) == 1 and c.source_id == c.gold[0] for c in cases)
    assert all(c.corpus == "g" for c in cases)
    # masking applied: no query leaks its gold artifact's name
    assert all(not eg._leaks_name(c.query, c.gold[0]) for c in cases)
    assert all(c.metadata["masked"] is True for c in cases)


def test_generate_cases_output_guard_drops_leaks():
    docs = {
        "alpha": {"name": "alpha", "description": "a long enough description of foo"}
    }

    def leaky(description, *, n):
        return ["please use alpha to do it", "do the foo thing now"]

    cases = eg.generate_cases(_skill_source(docs), k=2, query_generator=leaky)
    # the query that leaked the gold name ("alpha") is dropped; the clean one kept
    assert [c.query for c in cases] == ["do the foo thing now"]


def test_generate_cases_can_disable_masking():
    docs = {"alpha": {"name": "alpha", "description": "alpha does foo things for you"}}
    cases = eg.generate_cases(
        _skill_source(docs), k=1, query_generator=_echo_gen, mask_names=False
    )
    # with masking off, the name survives in the echoed description and is kept
    assert cases and "alpha" in cases[0].query
    assert cases[0].metadata["masked"] is False


def test_generate_cases_skips_short_descriptions_and_warns():
    docs = {"x": {"name": "x", "description": "too short"}}  # < min_chars
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = eg.generate_cases(
            _skill_source(docs), query_generator=_echo_gen, min_chars=20
        )
    assert cases == []
    assert any("skipped" in str(w.message) for w in caught)


def test_generate_cases_max_artifacts():
    docs = {
        f"a{i}": {"name": f"a{i}", "description": f"capability number {i} doing tasks"}
        for i in range(5)
    }
    cases = eg.generate_cases(
        _skill_source(docs), k=1, query_generator=_echo_gen, max_artifacts=2
    )
    assert {c.gold[0] for c in cases} == {"a0", "a1"}


# --------------------------------------------------------------------------- #
# abstention + build_eval_set
# --------------------------------------------------------------------------- #


def test_generate_abstention_cases():
    def gen(*, n, theme):
        return [f"off-topic request {i} about {theme}" for i in range(n)]

    cases = eg.generate_abstention_cases(3, generator=gen, theme="cooking")
    assert len(cases) == 3
    assert all(c.gold_is_none for c in cases)
    assert all(c.metadata["generator"] == "abstention" for c in cases)


def test_generate_abstention_cases_zero_is_empty():
    assert eg.generate_abstention_cases(0, generator=lambda **k: []) == []


def test_build_eval_set_hits_abstention_fraction():
    docs = {
        f"a{i}": {"name": f"a{i}", "description": f"capability number {i} doing tasks"}
        for i in range(5)
    }

    def qg(description, *, n):
        return [description]  # one gold case per artifact

    def ag(*, n, theme):
        return [f"unsupported {i}" for i in range(n)]

    cases = eg.build_eval_set(
        _skill_source(docs),
        k=1,
        abstention_frac=0.25,
        query_generator=qg,
        abstention_generator=ag,
    )
    n_abstain = sum(c.gold_is_none for c in cases)
    # 5 gold -> ceil(0.25 * 5 / 0.75) = 2 abstain -> 2/7 >= 0.25
    assert n_abstain == 2 and len(cases) == 7
    assert n_abstain / len(cases) >= 0.25


# --------------------------------------------------------------------------- #
# Generated set is scorable by ir.eval (end-to-end, offline)
# --------------------------------------------------------------------------- #


def test_generated_set_is_scorable_end_to_end():
    docs = {
        "alpha": {"name": "alpha", "description": "alpha tool for foo widget tasks"},
        "beta": {"name": "beta", "description": "beta tool for bar gadget chores"},
    }
    source = _skill_source(docs)
    cases = eg.generate_cases(source, k=1, query_generator=_echo_gen, corpus_name="g")
    corpus = ir.build(source, store=CorpusStore.memory(), embedder="light")
    report = ev.evaluate_discovery(corpus, cases, mode="dense", primary_k=1)
    assert report.n_gold == 2
    # the masked-description query still retrieves its own artifact first
    assert report.retrieval.metrics["recall@1"] == pytest.approx(1.0)


def test_save_load_with_signature_meta(tmp_path):
    docs = {"a": {"name": "a", "description": "a capability doing foo tasks for you"}}
    source = _skill_source(docs)
    cases = eg.build_eval_set(
        source,
        k=1,
        abstention_frac=0.0,
        query_generator=_echo_gen,
    )
    path = tmp_path / "cases.jsonl"
    ev.save_cases(cases, path, meta={"corpus_signature": eg.corpus_signature(source)})
    assert ev.load_cases(path) == cases


# --------------------------------------------------------------------------- #
# corpus_signature
# --------------------------------------------------------------------------- #


def test_corpus_signature_is_order_independent():
    docs = {
        "a": {"name": "a", "description": "x"},
        "b": {"name": "b", "description": "y"},
    }
    src1 = _skill_source(docs)
    src2 = _skill_source(dict(reversed(list(docs.items()))))
    assert eg.corpus_signature(src1) == eg.corpus_signature(src2)


def test_corpus_signature_changes_with_membership():
    base = {"a": {"name": "a", "description": "x"}}
    more = {
        "a": {"name": "a", "description": "x"},
        "b": {"name": "b", "description": "y"},
    }
    assert eg.corpus_signature(_skill_source(base)) != eg.corpus_signature(
        _skill_source(more)
    )


# --------------------------------------------------------------------------- #
# _parse_lines — strips real markers, never meaningful leading characters
# --------------------------------------------------------------------------- #


def test_parse_lines_preserves_leading_tokens():
    text = "\n".join(
        [
            "3D modeling help",
            "-9 degrees, what should I wear?",
            ".env file handling",
            "24/7 monitoring setup",
            "2024 tax question",
        ]
    )
    assert eg._parse_lines(text) == [
        "3D modeling help",
        "-9 degrees, what should I wear?",
        ".env file handling",
        "24/7 monitoring setup",
        "2024 tax question",
    ]


def test_parse_lines_still_strips_real_markers():
    assert eg._parse_lines("1. a\n2) b\n- c\n* d\n• e") == ["a", "b", "c", "d", "e"]


# --------------------------------------------------------------------------- #
# Masking — multi-word / whitespace / degenerate / self-referential
# --------------------------------------------------------------------------- #


def test_mask_name_whitespace_and_separator_tolerant():
    assert eg.mask_name("the data sync job", "data-sync") == "the this capability job"
    assert eg.mask_name("the data  sync job", "data sync") == "the this capability job"
    assert eg.mask_name("run datasync now", "data-sync") == "run this capability now"


def test_leaks_name_multiword_reordered_tokens():
    # bag-of-words: reordered/separated name tokens count as a leak
    assert eg._leaks_name("sync my data now", "data-sync")
    assert eg._leaks_name("i need data and a sync", "data sync")
    # a single shared token is NOT a leak (legitimate content overlap)
    assert not eg._leaks_name("just sync my files", "data-sync")


def test_mask_name_drops_degenerate_short_tokens():
    # a 1-char / separator-only name must not mask stray letters
    assert eg.mask_name("an example with x here", "-x") == "an example with x here"
    # but a real 2-char name is still masked
    assert eg.mask_name("the ci runs", "ci") == "the this capability runs"


def test_mask_name_avoids_self_referential_placeholder():
    out = eg.mask_name("The Capability is great", "capability")
    assert "this capability" not in out.lower()
    assert eg._ALT_PLACEHOLDER in out


# --------------------------------------------------------------------------- #
# _default_describe / _name_of fallbacks
# --------------------------------------------------------------------------- #


def test_default_describe_fallbacks():
    assert eg._default_describe({"description": "d", "text": "t"}) == "d"
    assert eg._default_describe({"text": "only text here"}) == "only text here"
    assert eg._default_describe({"name": "n", "summary": "s"}) == "n\ns"  # joined strs
    assert eg._default_describe("bare string") == "bare string"
    assert eg._default_describe({"description": "  ", "text": "fallback"}) == "fallback"


def test_name_of_coerces_missing_or_bad_name():
    assert eg._name_of("id1", {"name": None}) == "id1"
    assert eg._name_of("id1", {"name": ""}) == "id1"
    assert eg._name_of("id1", {"name": 42}) == "id1"
    assert eg._name_of("id1", {"name": "real"}) == "real"
    assert eg._name_of("id1", "bare") == "id1"


# --------------------------------------------------------------------------- #
# Validation + determinism + resilience
# --------------------------------------------------------------------------- #


def test_generate_cases_rejects_bad_k():
    src = _skill_source({"a": {"name": "a", "description": "x" * 30}})
    with pytest.raises(ValueError):
        eg.generate_cases(src, k=0, query_generator=_echo_gen)


def test_build_eval_set_rejects_out_of_range_frac():
    src = _skill_source({"a": {"name": "a", "description": "x" * 30}})
    with pytest.raises(ValueError):
        eg.build_eval_set(src, abstention_frac=1.0, query_generator=_echo_gen)
    with pytest.raises(ValueError):
        eg.build_eval_set(src, abstention_frac=-0.1, query_generator=_echo_gen)


def test_generate_cases_max_artifacts_is_deterministic_sorted():
    docs = {
        k: {"name": k, "description": f"{k} capability doing tasks here"}
        for k in ["zeta", "alpha", "mu"]
    }
    cases = eg.generate_cases(
        _skill_source(docs), k=1, query_generator=_echo_gen, max_artifacts=2
    )
    # sorted-id subset (alpha, mu), NOT the insertion-order first two (zeta, alpha)
    assert sorted({c.gold[0] for c in cases}) == ["alpha", "mu"]


def test_generate_cases_skips_on_generator_exception():
    docs = {
        "good": {"name": "good", "description": "good capability for foo tasks here"},
        "bad": {"name": "bad", "description": "bad capability triggering boom now ok"},
    }

    def flaky(description, *, n):
        if "boom" in description:
            raise RuntimeError("kaboom")
        return [description]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cases = eg.generate_cases(_skill_source(docs), k=1, query_generator=flaky)
    assert {c.gold[0] for c in cases} == {"good"}
    assert any("bad" in str(w.message) and "failed" in str(w.message) for w in caught)


def test_build_eval_set_forwards_mask_names_false():
    docs = {"alpha": {"name": "alpha", "description": "alpha does foo things for you"}}
    cases = eg.build_eval_set(
        _skill_source(docs),
        k=1,
        abstention_frac=0.0,
        query_generator=_echo_gen,
        mask_names=False,
    )
    assert cases and cases[0].metadata["masked"] is False
    assert "alpha" in cases[0].query


# --------------------------------------------------------------------------- #
# Default oa-backed generators — prompt assembly (skipped if oa absent)
# --------------------------------------------------------------------------- #


def test_oa_prompts_substitute_placeholders():
    oa = pytest.importorskip("oa")
    # prompt-only (prompt_func=None): assert on the ASSEMBLED PROMPT, not the
    # generate() wrapper (which, with prompt_func=None, would iterate the string).
    bt = oa.prompt_function(eg.BACKTRANSLATION_PROMPT, name="bt", prompt_func=None)
    prompt = bt(description="DESC_MARKER", n=4)
    assert "DESC_MARKER" in prompt and "Write 4 natural" in prompt
    ab = oa.prompt_function(eg.ABSTENTION_PROMPT, name="ab", prompt_func=None)
    assert "THEME_MARKER" in ab(theme="THEME_MARKER", n=2)


def test_parse_lines_is_wired_as_egress():
    oa = pytest.importorskip("oa")

    def fake_llm(*args, **kwargs):
        return "1. foo\n- bar"

    fn = oa.prompt_function(
        "x {description} {n}", egress=eg._parse_lines, prompt_func=fake_llm
    )
    assert fn(description="d", n=2) == ["foo", "bar"]


# --------------------------------------------------------------------------- #
# CLI eval-gen glue (build_eval_set stubbed; offline)
# --------------------------------------------------------------------------- #


def test_cli_eval_gen_glue(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("IR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IR_CACHE_DIR", str(tmp_path / "cache"))
    from ir import cli

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha content about widgets and gadgets here")
    cli.register("notes", "files", root=str(docs), pattern=r".*\.md$")

    captured = {}

    def fake_build(source, **kw):
        captured.update(kw)
        return [ev.DiscoveryCase("q1", gold=("a.md",)), ev.DiscoveryCase("q2", gold=())]

    # cli.eval_gen imports build_eval_set from ir.eval_gen at call time
    monkeypatch.setattr(eg, "build_eval_set", fake_build)

    out_path = tmp_path / "cases.jsonl"
    msg = cli.eval_gen("notes", str(out_path), k=3, max_artifacts="5")
    assert "2 cases" in msg and "1 gold" in msg and "1 abstention" in msg
    assert captured["k"] == 3 and captured["max_artifacts"] == 5  # int-cast + forwarded
    assert captured["corpus_name"] == "notes"

    loaded = ev.load_cases(out_path)
    assert len(loaded) == 2
    header = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])[
        "__meta__"
    ]
    assert header["corpus"] == "notes" and "corpus_signature" in header


def test_eval_gen_reachable_via_ir_namespace():
    assert hasattr(ir, "eval_gen")
    assert ir.eval_gen.generate_cases is eg.generate_cases
