"""Idempotent background-work runner — ``ir maintain`` (issue #58).

The *executing* half of ir's maintenance story (the declarative half is
:mod:`ir.policy`). :func:`maintain` reads each corpus's resolved
:class:`~ir.policy.MaintenancePolicy` and does the work that is **due**:

- an **incremental rebuild** when reindex is due (``source-change`` is always
  due — the build is a near-no-op when nothing changed; ``interval`` only when
  stale; ``manual`` never);
- when ``synopsis.enabled``, the rebuild's strategy is wrapped in
  :func:`ir.with_synopsis`, so new / changed artifacts gain an LLM synopsis —
  and because that build may call an LLM, it is run **only inside the policy's
  downtime window**.

It is safe to call as often as a scheduler likes: it no-ops when nothing is due,
and records ``last_maintained`` so interval policies converge. ir runs the work;
*scheduling* it is external — a cron / launchd entry calls ``ir maintain --all``
every N minutes, and the downtime window lives in the policy (data), not in ir.

:mod:`ir.schedule` installs that entry for you (``ir schedule``) and reports on it;
it writes a definition and exits, so the executor is still cron / launchd::

    # what `ir schedule --every 15m` writes on a cron machine:
    */15 * * * *  <python> -m ir maintain --all >> ~/.cache/ir/maintain.log 2>&1

Because a scheduled run has nobody watching it, :func:`maintain` is fault-isolated
per corpus: one corpus failing is recorded on its own result and the sweep carries
on, rather than aborting and leaving the corpora after it silently unmaintained.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import registry
from .config import cache_dir
from .index import build, open_corpus
from .policy import in_downtime, is_reindex_due, resolve_policy


#: A lock older than this is assumed to belong to a run that was killed rather
#: than one still working. Generous, because a first full build of a large corpus
#: legitimately takes a long time and reclaiming a *live* lock is the bad outcome.
DFLT_STALE_LOCK_AFTER = timedelta(hours=6)


class MaintenanceBusy(RuntimeError):
    """Another maintenance run holds the lock."""


def lock_path() -> Path:
    """Where the single-run lock lives (regenerable, so the cache dir)."""
    return cache_dir() / "maintain.lock"


def _lock_holder_is_live(path: Path, stale_after: timedelta) -> bool:
    """Whether the process that wrote the lock at *path* is plausibly still going."""
    try:
        pid = int(path.read_text(encoding="utf-8").split("\n", 1)[0].strip() or 0)
    except (OSError, ValueError):
        pid = 0
    # os.kill(pid, 0) is a liveness probe on POSIX only -- on Windows os.kill
    # terminates the process, so never probe there; fall back to age alone.
    if pid > 0 and os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # alive, just not ours
        except OSError:
            pass
        else:
            return True
    try:
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return False
    # Clamp: filesystem timestamp granularity (and clock skew) can put a
    # just-written mtime marginally in the future, and a negative age would then
    # compare as "younger than any threshold" -- making the lock immortal.
    return max(age, timedelta(0)) < stale_after


@contextmanager
def single_run(*, path: Path | None = None, stale_after: timedelta | None = None):
    """Hold the maintenance lock, or raise :class:`MaintenanceBusy`.

    Two maintenance runs on one corpus store are not safe: the packed store
    rewrites ``matrix``/``ids``/``metas`` as separate files, so interleaved
    writers can leave a matrix whose rows no longer line up with its ids — an
    index that answers confidently and wrongly, with nothing raised. launchd
    already runs one instance per label, but that does not cover cron (which
    happily stacks runs) or the manual ``ir maintain --all`` this tool's own
    menu suggests while the agent may be mid-run.

    A lock whose owning process is gone, or which is older than *stale_after*, is
    reclaimed — a killed run must not wedge maintenance forever.
    """
    path = Path(path) if path is not None else lock_path()
    stale_after = DFLT_STALE_LOCK_AFTER if stale_after is None else stale_after
    for _ in range(3):  # bounded: reclaim, then retry the create exactly once more
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_holder_is_live(path, stale_after):
                raise MaintenanceBusy(
                    f"another ir maintain run is in progress (lock: {path}). "
                    "Wait for it, or delete that file if you are sure it is dead."
                ) from None
            try:
                path.unlink()
            except OSError:
                pass
            continue
        try:
            os.write(fd, f"{os.getpid()}\n{datetime.now().isoformat()}\n".encode())
        finally:
            os.close(fd)
        try:
            yield path
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        return
    raise MaintenanceBusy(f"could not take the maintenance lock at {path}")


@dataclass
class MaintenanceResult:
    """What :func:`maintain_corpus` did (or would do) for one corpus."""

    name: str
    ran: bool
    reason: str
    reindex: bool = False
    synopsis: bool = False
    records: int | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"{self.name}: FAILED ({self.error})"
        verb = "maintained" if self.ran else "skipped"
        bits = [
            b
            for b, on in (("reindex", self.reindex), ("synopsis", self.synopsis))
            if on
        ]
        what = "+".join(bits) or "-"
        recs = f", {self.records} records" if self.records is not None else ""
        return f"{self.name}: {verb} [{what}] ({self.reason}){recs}"


def _now(now: datetime | None) -> datetime:
    # Local clock: the downtime window is expressed in local hours, and interval
    # math stays consistent by using the same clock for both. Injectable for tests.
    return now if now is not None else datetime.now()


def _parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _synopsis_wrapped(source):
    """Return *source* with its strategy wrapped in :func:`ir.with_synopsis`."""
    from .synopsis import with_synopsis

    return dc_replace(source, indexing_strategy=with_synopsis(source.indexing_strategy))


def maintain_corpus(
    name: str, *, now: datetime | None = None, dry_run: bool = False, full: bool = True
) -> MaintenanceResult:
    """Do the due background work for one corpus (idempotent).

    Reads the corpus's resolved policy and its ``last_maintained`` time, decides
    whether a (synopsis-aware) reindex is due *and* permitted now, and — unless
    ``dry_run`` — runs the incremental build and records the run.
    """
    now = _now(now)
    policy = resolve_policy(registry.get(name))
    store = open_corpus(name).store  # lazy: opening never loads the model
    last = _parse_iso(store.get_maintenance_state().get("last_maintained"))

    due = is_reindex_due(policy, last, now)
    synopsis_on = policy.synopsis.enabled
    # A synopsis build may hit an LLM, so it is confined to the downtime window.
    blocked = synopsis_on and not in_downtime(policy, now)

    if not due or blocked:
        reason = (
            "synopsis build deferred to downtime window"
            if due and blocked
            else f"not due ({policy.reindex.on})"
        )
        return MaintenanceResult(name, False, reason, reindex=due, synopsis=synopsis_on)

    if dry_run:
        return MaintenanceResult(
            name, False, "dry-run (would reindex)", reindex=True, synopsis=synopsis_on
        )

    source = registry.source_for(name)
    if synopsis_on:
        source = _synopsis_wrapped(source)
    built = build(source, full=full)
    built.store.set_maintenance_state(
        {"last_maintained": now.isoformat(), "synopsis_enabled": synopsis_on}
    )
    return MaintenanceResult(
        name, True, "reindexed", reindex=True, synopsis=synopsis_on, records=len(built)
    )


def maintain(
    name: str | None = None,
    *,
    all: bool = False,
    now: datetime | None = None,
    dry_run: bool = False,
) -> list[MaintenanceResult]:
    """Run due background work for one corpus (*name*) or every registered one (*all*).

    With neither, defaults to all registered corpora. Returns one
    :class:`MaintenanceResult` per corpus considered.

    **A named corpus raises; a sweep records.** Asking for one corpus by name is a
    direct request, and swallowing its error would turn a typo'd name into a
    result that reads as success to any caller written before ``error`` existed.
    A sweep is the unattended case, where the opposite is true: one corpus
    failing must not leave every corpus after it stale with nothing to say so.
    """
    sweep = not (name and not all)
    names = list(registry.registered()) if sweep else [name]
    results = []
    for n in names:
        if not sweep:
            results.append(maintain_corpus(n, now=now, dry_run=dry_run))
            continue
        try:
            results.append(maintain_corpus(n, now=now, dry_run=dry_run))
        except Exception as exc:
            results.append(
                MaintenanceResult(
                    n, False, f"error: {exc}", error=f"{type(exc).__name__}: {exc}"
                )
            )
    return results
