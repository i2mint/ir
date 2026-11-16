"""Ledger provenance and replay through ir.eval (#136)."""

import pytest

import ir
from ir.store import CorpusStore
from ir.synonyms import (
    DEMOTED,
    SynonymEntry,
    add_synonyms,
    load_synonyms,
)
from ir.synonyms_eval import (
    HELPS,
    NO_EFFECT,
    UNVERIFIABLE,
    EntryVerdict,
    apply_verdicts,
    case_of,
    replay_entry,
    replay_ledger,
)

DOCS = {
    "d1": "acmecorp quarterly summary",
    "d2": "ac quarterly report ac ac",
    "d3": "bake a cake with flour",
}
PROV = {"miss_query": "ac", "found_ids": ["d1"]}


@pytest.fixture
def corpus():
    src = ir.CorpusSource.from_mapping(DOCS, name="led", strategy=ir.WholeText())
    return ir.build(src, store=CorpusStore.memory(), embedder="light")


def test_provenance_round_trips_through_the_ledger_file(tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    (e,) = load_synonyms(p)
    assert e.provenance == PROV and e.verified == 0 and e.status == "active"
    # defaults stay out of the file, so existing ledgers keep their shape
    add_synonyms(["x1", "y1"], path=p)
    assert "provenance" not in p.read_text().splitlines()[1]
    assert "status" not in p.read_text().splitlines()[1]


def test_demoted_entries_are_kept_but_not_loaded(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"terms": ["aa", "bb"], "status": "demoted"}\n')
    assert load_synonyms(p) == []
    assert len(load_synonyms(p, include_demoted=True)) == 1


def test_case_of_needs_a_complete_provenance():
    assert case_of(SynonymEntry(("a1", "b1"))) is None
    assert case_of(SynonymEntry(("a1", "b1"), provenance={"miss_query": "q"})) is None
    c = case_of(SynonymEntry(("a1", "b1"), provenance=PROV))
    assert c.query == "ac" and c.gold == ("d1",)


def test_an_entry_that_found_the_gold_helps(corpus):
    v = replay_entry(corpus, SynonymEntry(("acmecorp", "ac"), provenance=PROV))
    assert v.verdict == HELPS and v.with_entry > v.without


def test_an_entry_that_does_not_apply_has_no_effect(corpus):
    v = replay_entry(corpus, SynonymEntry(("zzz", "yyy"), provenance=PROV))
    assert v.verdict == NO_EFFECT and v.with_entry == v.without


def test_entry_without_provenance_is_unverifiable(corpus):
    assert replay_entry(corpus, SynonymEntry(("a1", "b1"))).verdict == UNVERIFIABLE


def test_replay_apply_counts_wins_and_demotes_the_rest(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    add_synonyms(["zzz", "yyy"], provenance=PROV, path=p)
    add_synonyms(["no1", "no2"], path=p)  # no provenance
    verdicts = replay_ledger(corpus, path=p, apply=True)
    assert [v.verdict for v in verdicts] == [HELPS, NO_EFFECT, UNVERIFIABLE]
    by = {e.terms[0]: e for e in load_synonyms(p, include_demoted=True)}
    assert by["acmecorp"].verified == 1 and by["acmecorp"].status == "active"
    assert by["zzz"].status == DEMOTED
    assert by["no1"].status == "active" and by["no1"].verified == 0
    # the demoted one no longer expands searches; the survivor does
    assert [e.terms[0] for e in load_synonyms(p)] == ["acmecorp", "no1"]


def test_replay_without_apply_does_not_touch_the_file(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["zzz", "yyy"], provenance=PROV, path=p)
    before = p.read_text()
    replay_ledger(corpus, path=p)
    assert p.read_text() == before


def _line_of(p, terms0):
    return next(ln for ln in p.read_text().splitlines() if f'"{terms0}"' in ln)


def test_apply_preserves_comments_malformed_lines_and_unknown_keys(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    p.write_text(
        p.read_text()
        + "# a hand comment\n"
        + '{"terms": "broken"}\n'
        + '{"terms": ["zzz", "yyy"], "custom_key": 7, "provenance": '
        + '{"miss_query": "ac", "found_ids": ["d1"]}}\n'
    )
    replay_ledger(corpus, path=p, apply=True)
    lines = p.read_text().splitlines()
    assert "# a hand comment" in lines and '{"terms": "broken"}' in lines
    zzz = _line_of(p, "zzz")
    assert '"custom_key": 7' in zzz and '"status": "demoted"' in zzz


def test_wrong_corpus_or_missing_gold_is_unverifiable_not_demoted(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    gone = {"miss_query": "ac", "found_ids": ["not-in-this-corpus"]}
    other = {**PROV, "corpus": "some-other-corpus"}
    add_synonyms(["zzz", "yyy"], provenance=gone, path=p)
    add_synonyms(["ppp", "qqq"], provenance=other, path=p)
    verdicts = replay_ledger(corpus, path=p, apply=True)
    assert [v.verdict for v in verdicts] == [UNVERIFIABLE, UNVERIFIABLE]
    assert len(load_synonyms(p)) == 2  # nothing demoted


def test_no_effect_demotes_only_an_entry_that_never_won():
    won = SynonymEntry(("a1", "b1"), verified=2)
    never = SynonymEntry(("c1", "d1"))
    changes = apply_verdicts(
        [EntryVerdict(won, NO_EFFECT), EntryVerdict(never, NO_EFFECT)]
    )
    assert [(o.terms[0], n.status) for o, n in changes] == [("c1", DEMOTED)]
    hurts = apply_verdicts([EntryVerdict(won, "hurts")])
    assert hurts[0][1].status == DEMOTED


def test_a_demoted_entry_that_wins_again_is_reactivated(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"terms": ["acmecorp", "ac"], "status": "demoted", "provenance": '
        '{"miss_query": "ac", "found_ids": ["d1"]}}\n'
    )
    replay_ledger(corpus, path=p, apply=True)
    (e,) = load_synonyms(p)
    assert e.status == "active" and e.verified == 1


def test_entries_with_apply_and_formulate_are_refused(corpus):
    e = SynonymEntry(("acmecorp", "ac"), provenance=PROV)
    with pytest.raises(ValueError, match="entries"):
        replay_ledger(corpus, entries=[e], apply=True)
    with pytest.raises(ValueError, match="formulate"):
        replay_entry(corpus, e, formulate=lambda q: q)


def test_duplicate_identical_lines_each_get_their_verdict(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    replay_ledger(corpus, path=p, apply=True)
    assert [e.verified for e in load_synonyms(p)] == [1, 1]


def test_cli_replay_dry_run_then_apply(corpus, tmp_path, monkeypatch):
    from ir import cli

    p = tmp_path / "s.jsonl"
    monkeypatch.setenv("IR_SYNONYMS", str(p))
    monkeypatch.setattr(cli, "_open_ready", lambda name: corpus)
    add_synonyms(["acmecorp", "ac"], provenance=PROV, path=p)
    add_synonyms(["zzz", "yyy"], provenance=PROV, path=p)
    before = p.read_text()
    out = cli.synonyms_replay("led")
    assert "helps" in out and "no_effect" in out and "would change" in out
    assert p.read_text() == before
    assert "updated" in cli.synonyms_replay("led", apply=True)
    assert [e.terms[0] for e in load_synonyms(p)] == ["acmecorp"]


def test_cli_synonyms_add_records_provenance(tmp_path, monkeypatch):
    from ir import cli

    p = tmp_path / "s.jsonl"
    monkeypatch.setenv("IR_SYNONYMS", str(p))
    cli.synonyms_add("acmecorp, ac", miss_query="ac", found="d1, d2", corpus="led")
    (e,) = load_synonyms(p)
    assert e.provenance == {
        "miss_query": "ac",
        "found_ids": ["d1", "d2"],
        "corpus": "led",
    }
    with pytest.raises(SystemExit, match="go together"):
        cli.synonyms_add("a1, b1", miss_query="ac")


def test_bad_verified_value_skips_the_line_not_the_search(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"terms": ["aa", "bb"], "verified": "x"}\n'
        '{"terms": ["cc", "dd"], "verified": [1]}\n'
        '{"terms": ["ee", "ff"]}\n'
    )
    assert [e.terms[0] for e in load_synonyms(p)] == ["ee"]


def test_exotic_line_separators_survive_a_patch(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    add_synonyms(["zzz", "yyy"], note="line sep", provenance=PROV, path=p)
    replay_ledger(corpus, path=p, apply=True)
    (e,) = load_synonyms(p, include_demoted=True)
    assert e.note == "line sep" and e.status == DEMOTED


def test_no_change_means_no_rewrite(corpus, tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"terms": ["zzz", "yyy"], "status": "demoted", "provenance": '
        '{"miss_query": "ac", "found_ids": ["d1"]}}\n'
    )
    before = p.stat().st_mtime_ns
    replay_ledger(corpus, path=p, apply=True)
    assert p.stat().st_mtime_ns == before
