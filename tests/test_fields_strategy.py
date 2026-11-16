"""The ``Fields`` strategy: a record's named text fields (#140)."""

import pytest

import ir
from ir.strategy import (
    Fields,
    balanced_windows,
    fold_text,
    strategy_from_spec,
    strategy_to_spec,
)
from ir.index import _strategy_id
from ir.store import CorpusStore

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


def test_normalize_by_name_and_by_callable():
    rec = {"name": "Crème Brûlée"}
    assert _texts(Fields([], head=["name"], normalize="fold").decompose("r", rec)) == [
        ("record", "creme brulee")
    ]
    plan = Fields([], head=["name"], normalize=str.upper).decompose("r", rec)
    assert _texts(plan) == [("record", "CRÈME BRÛLÉE")]
    assert fold_text("É") == "e"
    with pytest.raises(ValueError, match="unknown normalizer"):
        Fields(["a"], normalize="nope")
    with pytest.raises(ValueError, match="layout"):
        Fields(["a"], layout="x")


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
    a = Fields(["about"], head=["name"], normalize="fold")
    clone = strategy_from_spec(strategy_to_spec(a))
    assert isinstance(clone, Fields) and clone.fields == ("about",)
    assert clone.head == ("name",) and clone.normalize == "fold"
    assert _strategy_id(a) == _strategy_id(clone)
    assert _strategy_id(a) != _strategy_id(Fields(["about", "more"], head=["name"]))


def test_fields_beats_whole_record_split_on_a_cross_field_query():
    """A word in the name and a word in the place only score together in 'record'."""
    docs = {
        "r1": {"name": "Alpha Hotel", "place": "Lisbon", "about": "rooms"},
        "r2": {"name": "Alpha Hotel", "place": "Oslo", "about": "rooms"},
        "r3": {"name": "Beta Hotel", "place": "Lisbon", "about": "rooms"},
    }
    ranks = {}
    for layout in ("record", "fields"):
        src = ir.CorpusSource.from_mapping(
            docs,
            name=f"fld_{layout}",
            strategy=Fields(["about"], head=["name", "place"], layout=layout),
        )
        corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
        hits = corpus.search("alpha lisbon", mode="lexical", k=3)
        ranks[layout] = [h.artifact_id for h in hits]
    assert ranks["record"][0] == "r1"
