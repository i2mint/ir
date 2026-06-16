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

    # crontab: try the queue every 15 minutes; ir decides what is actually due.
    */15 * * * *  ir maintain --all >> ~/.cache/ir/maintain.log 2>&1
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from datetime import datetime
from typing import Any

from . import registry
from .index import build, open_corpus
from .policy import in_downtime, is_reindex_due, resolve_policy


@dataclass
class MaintenanceResult:
    """What :func:`maintain_corpus` did (or would do) for one corpus."""

    name: str
    ran: bool
    reason: str
    reindex: bool = False
    synopsis: bool = False
    records: int | None = None

    def __str__(self) -> str:
        verb = "maintained" if self.ran else "skipped"
        bits = [b for b, on in (("reindex", self.reindex), ("synopsis", self.synopsis)) if on]
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
    """
    if name and not all:
        names = [name]
    else:
        names = list(registry.registered())
    return [maintain_corpus(n, now=now, dry_run=dry_run) for n in names]
