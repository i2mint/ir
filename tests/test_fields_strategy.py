"""The ``Fields`` strategy: a record's named text fields (#140)."""

import pytest

import ir
from ir.strategy import (
    Fields,
    balanced_windows,
    strategy_from_spec,
    strategy_to_spec,
)
from ir.index import _strategy_id
from ir.store import CorpusStore


def _upper(text):
    return text.upper()


REC = {"name": "Louvre", "place": "Paris", "about": "A museum of art.", "id": 7}


def _texts(plan):
    return [(s.kind, s.text) for s in plan.surfaces]


def test_record_layout_prefixes_head_to_the_body():
    plan = Fields(["about"], head=["name", "place"]).decompose("r", REC)
    assert _texts(plan) == [("record", "Louvre\nParis\nA museum of art.")]


def test_record_layout_repeats_head_on_every_window():
    rec = {"name": "N", "about": "x" * 50}
    plan = Fields(["about"], head=["name"], max_chars=30, overlap=0).decompose("r", rec)
    assert len(plan.surfaces) == 2
    assert all(s.text.startswith("N\n") for s in plan.surfaces)
    assert [s.metadata["n_chunks"] for s in plan.surfaces] == [2, 2]


def test_fields_layout_is_one_surface_per_field_named_by_kind():
    plan = Fields(["about"], head=["name", "place"], layout="fields").decompose(
        "r", REC
    )
    assert _texts(plan) == [
        ("name", "Louvre"),
        ("place", "Paris"),
        ("about", "A museum of art."),
    ]


def test_missing_and_empty_fields_are_skipped():
    plan = Fields(["about", "gone"], head=["name"]).decompose(
        "r", {"name": "N", "about": "", "gone": None}
    )
    assert _texts(plan) == [("record", "N")]
    assert Fields(["about"]).decompose("r", {"about": "  "}).surfaces == []
    assert Fields(["about"]).decompose("r", "not a mapping").surfaces == []


def test_normalize_callable_applies_to_every_text():
    plan = Fields([], head=["name"], normalize=str.upper).decompose(
        "r", {"name": "Crème Brûlée"}
    )
    assert _texts(plan) == [("record", "CRÈME BRÛLÉE")]


def test_normalizer_identity_tracks_which_callable():
    assert _strategy_id(Fields(["a"], normalize=str.upper)) != _strategy_id(
        Fields(["a"], normalize=str.lower)
    )


def test_constructor_validates():
    with pytest.raises(ValueError, match="layout"):
        Fields(["a"], layout="x")
    with pytest.raises(TypeError, match="bare string"):
        Fields("about")
    with pytest.raises(TypeError, match="bare string"):
        Fields(["a"], head="name")
    with pytest.raises(ValueError, match="overlap"):
        Fields(["a"], max_chars=10, overlap=10)
    with pytest.raises(ValueError, match="overlap"):
        Fields(["a"], max_chars=0, overlap=0)
    with pytest.raises(TypeError):
        Fields(["a"], max_chars="12")


def test_a_key_in_head_and_fields_is_read_once():
    f = Fields(["name", "about"], head=["name"])
    assert f.reads == ("name", "about")
    plan = Fields(["name"], head=["name"], layout="fields").decompose("r", REC)
    assert [s.kind for s in plan.surfaces] == ["name"]


def test_fields_layout_windows_carry_chunk_metadata():
    plan = Fields(["about"], layout="fields", max_chars=30, overlap=0).decompose(
        "r", {"about": "x" * 50}
    )
    assert [s.metadata["n_chunks"] for s in plan.surfaces] == [2, 2]


@pytest.mark.parametrize("length", [11, 12, 19, 25, 100])
@pytest.mark.parametrize("overlap", [0, 3])
def test_balanced_windows_are_equal_and_cover_the_text(length, overlap):
    text = "".join(chr(97 + i % 26) for i in range(length))
    wins = balanced_windows(text, size=10, overlap=overlap)
    widths = [len(w) for w in wins]
    # equal up to rounding: only the last may be shorter, by fewer than n chars
    assert max(widths) <= 10 and max(widths) - min(widths) < len(wins)
    assert widths[:-1] == [widths[0]] * (len(wins) - 1)
    # consecutive windows overlap by exactly `overlap` chars and the ends are covered
    assert wins[0] == text[: len(wins[0])] and text.endswith(wins[-1])
    for a, b in zip(wins, wins[1:], strict=False):
        assert a[len(a) - overlap :] == b[:overlap] if overlap else True


def test_balanced_windows_no_sliver_and_guards():
    assert [len(w) for w in balanced_windows("a" * 11, size=10)] == [6, 5]
    assert balanced_windows("  ", size=10) == []
    with pytest.raises(ValueError):
        balanced_windows("abc", size=5, overlap=5)


def test_spec_round_trip_and_identity_track_the_fields():
    a = Fields(["about"], head=["name"], max_chars=500)
    clone = strategy_from_spec(strategy_to_spec(a))
    assert isinstance(clone, Fields) and clone.fields == ("about",)
    assert clone.head == ("name",) and clone.max_chars == 500
    assert _strategy_id(a) == _strategy_id(clone)
    assert _strategy_id(a) != _strategy_id(Fields(["about", "more"], head=["name"]))


def test_record_layout_scores_cross_field_queries_above_fields_layout():
    """A word in the name and a word in the place only score together in 'record'."""
    docs = {
        "r1": {"name": "Alpha Hotel", "place": "Lisbon", "about": "rooms"},
        "r2": {"name": "Alpha Hotel", "place": "Oslo", "about": "rooms"},
        "r3": {"name": "Beta Hotel", "place": "Lisbon", "about": "rooms"},
    }
    margin = {}
    for layout in ("record", "fields"):
        src = ir.CorpusSource.from_mapping(
            docs,
            name=f"fld_{layout}",
            strategy=Fields(["about"], head=["name", "place"], layout=layout),
        )
        corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
        hits = corpus.search("alpha lisbon", mode="lexical", k=3, per_artifact=True)
        score = {h.artifact_id: h.score for h in hits}
        assert max(score, key=score.get) == "r1"
        margin[layout] = score["r1"] / max(score["r2"], score["r3"])
    assert margin["record"] > margin["fields"]


def test_from_records_reindexes_when_a_read_field_changes():
    from ir.sources import CorpusSource

    def mk(about):
        return CorpusSource.from_records(
            name="rf",
            fetcher=lambda: [{"id": "x", "text": "t", "name": "n", "about": about}],
            strategy=Fields(["about"], head=["name"]),
        )

    one, two = mk("first"), mk("second")
    sig = one.change_signal
    assert sig("x", one.scope["x"]) != sig("x", two.scope["x"])


def test_normalizer_reference_round_trips_and_resolves():
    a = Fields([], head=["name"], normalize="test_fields_strategy:_upper")
    clone = strategy_from_spec(strategy_to_spec(a))
    assert clone.normalize == "test_fields_strategy:_upper"
    assert _strategy_id(a) == _strategy_id(clone)
    assert _texts(clone.decompose("r", {"name": "ab"})) == [("record", "AB")]


def test_unnamed_normalizer_needs_an_explicit_id():
    with pytest.warns(UserWarning, match="normalize_id"):
        Fields(["a"], normalize=lambda t: t)
    tagged = Fields(["a"], normalize=lambda t: t, normalize_id="v1")
    other = Fields(["a"], normalize=lambda t: t, normalize_id="v2")
    assert _strategy_id(tagged) != _strategy_id(other)
    with pytest.raises(ValueError, match="requires a callable"):
        Fields(["a"], normalize="test_fields_strategy:_upper", normalize_id="x")


def test_duplicate_keys_collapse():
    assert Fields(["a", "a"], head=["h", "h"]).reads == ("h", "a")


# ----- #146 follow-ups: safe fold, synopsis wrapper forwarding -------------- #


def test_fold_text_strips_latin_and_greek_accents_only():
    from ir.strategy import fold_text

    assert fold_text("Crème BRÛLÉE Straße") == "creme brulee strasse"
    assert fold_text("Αθήνα") == "αθηνα"
    assert fold_text("Đà Nẵng Łódź Øre Æsop") == "da nang lodz ore aesop"
    assert fold_text("Ｃｒèｍｅ") == "creme"
    for text in ("Crème", "e\u0301", "Ｃｒèｍｅ ß", "Αθήνα", "がぎ", "한국어"):
        assert fold_text(fold_text(text)) == fold_text(text)  # idempotent
    for untouched in ("がぎぱ", "한국어", "हिन्दी", "ไก่", "йё"):
        assert fold_text(untouched) == untouched
    # a decomposed accent on a Latin letter goes too; one after kana stays
    assert fold_text("e\u0301") == "e"
    assert fold_text("\u304b\u3099") == "\u304c"  # NFC composes; the mark is kept
    assert fold_text("") == ""


def test_fold_text_works_as_a_normalizer_reference_and_a_formulator():
    f = Fields([], head=["name"], normalize="ir.strategy:fold_text")
    plan = f.decompose("r", {"name": "Crème"})
    assert [s.text for s in plan.surfaces] == ["creme"]
    from ir.strategy import fold_text

    # the same function on the query side makes the folded index findable
    src = ir.CorpusSource.from_mapping(
        {"r": {"name": "Crème Brûlée"}},
        name="fold",
        strategy=Fields([], head=["name"], normalize=fold_text),
    )
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    miss = ir.search(corpus, "Crème", mode="lexical")
    hit = ir.search(corpus, "Crème", mode="lexical", formulate=fold_text)
    assert [h.artifact_id for h in hit] == ["r"] and not miss


def test_with_synopsis_forwards_reads_and_synopsis_fields():
    wrapped = ir.with_synopsis(
        Fields(["about"], head=["name"]), synthesize=lambda raw: "summary"
    )
    assert wrapped.reads == ("name", "about")
    from ir.sources import CorpusSource

    src = CorpusSource.from_records(
        name="ws",
        fetcher=lambda: [{"id": "x", "text": "t", "name": "n", "about": "a"}],
        strategy=wrapped,
    )
    other = CorpusSource.from_records(
        name="ws",
        fetcher=lambda: [{"id": "x", "text": "t", "name": "n", "about": "b"}],
        strategy=wrapped,
    )
    assert src.change_signal("x", src.scope["x"]) != other.change_signal(
        "x", other.scope["x"]
    )
    assert Fields(["about"], head=["name"]).synopsis_fields == ("name", "about")
    from ir.synopsis import _default_synthesizer

    synth = _default_synthesizer(Fields(["about"], head=["name"]))
    assert synth.synthesizer_id.endswith(":name,about")  # the strategy's fields
