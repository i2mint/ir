"""Regression tests for incremental-maintenance edge cases."""

import os
import time

import pytest

import ir
from ir.base import storage_key
from ir.store import CorpusStore


def test_strategy_change_triggers_reindex():
    # Same content, different strategy -> must re-decompose (not skipped).
    text = "\n\n".join(f"paragraph {i} with several words here" for i in range(30))
    docs = {"d": {"text": text}}
    store = CorpusStore.memory()

    ir.build(
        ir.CorpusSource.from_mapping(docs, name="m", strategy=ir.WholeText()),
        store=store,
        embedder="light",
    )
    assert len(store) == 1  # WholeText -> one surface

    ir.build(
        ir.CorpusSource.from_mapping(
            docs, name="m", strategy=ir.Chunked(chunk_size=120, overlap=20)
        ),
        store=store,
        embedder="light",
    )
    assert len(store) > 1  # Chunked -> many surfaces, despite unchanged content
    assert "Chunked" in store.get_ledger_entry(storage_key("d"))["strategy_id"]


def test_from_skills_fetcher_injection():
    fake = [
        {"name": "alpha", "description": "manage alpha widgets", "parent": "pkgA"},
        {"name": "beta", "description": "manage beta gadgets", "parent": "pkgB"},
    ]
    src = ir.CorpusSource.from_skills(fetcher=lambda: fake)
    assert set(src.scope) == {"alpha", "beta"}
    corpus = ir.build(src, store=CorpusStore.memory(), embedder="light")
    assert len(corpus) == 2
    assert ir.search(corpus, "alpha widgets", k=1)[0].metadata["name"] == "alpha"


def test_sweep_is_fault_isolated_per_corpus(monkeypatch):
    # A scheduled run has nobody watching it: one corpus blowing up must not
    # leave the corpora after it silently unmaintained (the failure that left a
    # real corpus stale for months while the sweep "succeeded").
    from ir import maintenance

    monkeypatch.setattr(
        maintenance.registry, "registered", lambda: {"good": {}, "bad": {}, "also": {}}
    )

    def fake_maintain_corpus(name, **kwargs):
        if name == "bad":
            raise RuntimeError("source vanished")
        return maintenance.MaintenanceResult(name, True, "reindexed", reindex=True)

    monkeypatch.setattr(maintenance, "maintain_corpus", fake_maintain_corpus)

    results = maintenance.maintain(all=True)
    assert [r.name for r in results] == ["good", "bad", "also"]
    assert [r.ran for r in results] == [True, False, True]

    failed = results[1]
    assert failed.error == "RuntimeError: source vanished"
    assert "FAILED" in str(failed) and "source vanished" in str(failed)


def test_a_named_corpus_still_raises(monkeypatch):
    # Fault isolation is right for the unattended sweep and wrong as the only
    # behaviour of a public single-name API: a typo'd name used to raise a
    # KeyError telling you to register it, and every pre-existing programmatic
    # caller (written before `error` existed) would read the swallowed version as
    # success.
    from ir import maintenance

    monkeypatch.setattr(maintenance.registry, "registered", lambda: {"good": {}})

    def boom(name, **kwargs):
        raise KeyError(f"corpus {name!r} is not registered")

    monkeypatch.setattr(maintenance, "maintain_corpus", boom)
    with pytest.raises(KeyError):
        maintenance.maintain("typo-d-name")
    # ...while the sweep over the same failing corpus records instead.
    assert maintenance.maintain(all=True)[0].error


def test_single_run_lock_excludes_a_second_run(tmp_path):
    # Two maintenance runs on one corpus store can interleave writes to the
    # packed store's separate matrix/ids/metas files, leaving a matrix whose rows
    # no longer line up with its ids -- an index that answers confidently wrong.
    from ir.maintenance import MaintenanceBusy, single_run

    lock = tmp_path / "maintain.lock"
    with single_run(path=lock):
        assert lock.exists()
        with pytest.raises(MaintenanceBusy):
            with single_run(path=lock):
                raise AssertionError("second run must not get the lock")
    assert not lock.exists()  # released on the way out


@pytest.mark.skipif(
    os.name != "posix", reason="pid liveness probing is POSIX-only (os.kill kills on Windows)"
)
def test_single_run_lock_reclaims_a_dead_holder(tmp_path):
    # A killed run must not wedge maintenance forever. On POSIX the owning pid is
    # probed directly, so a dead holder is reclaimed however fresh its lockfile.
    from ir.maintenance import single_run

    lock = tmp_path / "maintain.lock"
    lock.write_text("999999999\n2020-01-01T00:00:00\n", encoding="utf-8")
    with single_run(path=lock):
        assert lock.read_text(encoding="utf-8").startswith(str(os.getpid()))


def test_single_run_lock_reclaims_a_stale_lock(tmp_path):
    # Where the pid cannot be probed (Windows) or is unreadable, age is the only
    # signal -- so that path has to work on its own.
    from datetime import timedelta

    from ir.maintenance import single_run

    lock = tmp_path / "maintain.lock"
    lock.write_text("not-a-pid\n", encoding="utf-8")
    # Backdate rather than lean on sub-second timing: a just-written mtime can
    # land marginally in the *future* on some filesystems.
    old_time = time.time() - 3600
    os.utime(lock, (old_time, old_time))
    with single_run(path=lock, stale_after=timedelta(minutes=1)):
        assert lock.exists()


def test_a_future_dated_lock_is_not_immortal(tmp_path):
    # A negative age must not compare as "younger than any threshold".
    from datetime import timedelta

    from ir.maintenance import single_run

    lock = tmp_path / "maintain.lock"
    lock.write_text("not-a-pid\n", encoding="utf-8")
    ahead = time.time() + 3600
    os.utime(lock, (ahead, ahead))
    with single_run(path=lock, stale_after=timedelta(seconds=0)):
        assert lock.exists()


def test_maintain_cli_prints_the_report_on_stdout_even_when_a_corpus_fails(
    monkeypatch, capsys
):
    # `ir maintain --all > report.txt` must not lose the corpora that succeeded
    # just because one did not. Only the summary rides SystemExit (whose job is
    # the non-zero exit code a scheduler records).
    from ir import cli, maintenance

    monkeypatch.setattr(
        maintenance,
        "maintain",
        lambda **kwargs: [
            maintenance.MaintenanceResult("good", True, "reindexed", records=231),
            maintenance.MaintenanceResult("bad", False, "error: boom", error="E: boom"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.maintain(all=True)
    out = capsys.readouterr().out
    assert "good: maintained" in out and "231 records" in out
    assert "bad: FAILED" in out
    assert "1 of 2 corpora failed" in str(excinfo.value)


def test_maintain_cli_reports_a_concurrent_run_instead_of_racing(monkeypatch):
    from contextlib import contextmanager

    from ir import cli, maintenance

    @contextmanager
    def busy():
        raise maintenance.MaintenanceBusy("another ir maintain run is in progress")
        yield  # pragma: no cover

    monkeypatch.setattr(maintenance, "single_run", busy)
    assert "skipped:" in cli.maintain(all=True)
