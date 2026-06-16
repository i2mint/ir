"""Per-corpus policy — how a corpus is segmented/stored and what background work it gets.

This is the *declarative* half of ir's maintenance story (issue #58): a corpus's
policy lives as **data** in its registry entry, and ir resolves an effective
policy by layering ``entry`` over per-``kind`` defaults over a global default.
An idempotent :func:`ir.maintenance.maintain` reads the policy and does the due
work; *scheduling* that work (cron / launchd, or an orchestration layer's budget
governor) stays **external** — ir never imports an orchestration layer, and there
is no global ``Settings`` singleton (policy is per-corpus data, injected, not a
process-wide mutable). The light path stays a one-liner: every field has a smart
default, so ``ir build skills`` needs no policy at all.

The three policy axes:

- **reindex** — *when* to rebuild: ``"source-change"`` (default; rebuild is a
  near-no-op when nothing changed, so it is always safe to run), ``"interval"``
  (only when older than ``every_hours``), or ``"manual"`` (never automatic).
- **synopsis** — *whether/when* to attach LLM synopses (the expensive,
  off-by-default work): only the ``recent`` slice, only during ``downtime_hours``.
  Synopsis is realized as a strategy wrapper (:func:`ir.with_synopsis`), so it is
  incremental — only new/changed artifacts are synthesized.
- **storage** — the persistence backend. Today only ``"local"`` (the file store);
  a ``vd.Collection`` backend is the documented future seam (issue #28).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

# --------------------------------------------------------------------------- #
# Policy data model (immutable; pure data — no I/O, no model loading)
# --------------------------------------------------------------------------- #

#: Accepted ``reindex.on`` triggers.
REINDEX_TRIGGERS = ("source-change", "interval", "manual")

#: Storage backends ir can resolve today. ``"vd"`` (vd.Collection) is the
#: documented future backend, gated on issue #28.
STORAGE_BACKENDS = ("local",)


@dataclass(frozen=True)
class ReindexPolicy:
    """When to (incrementally) rebuild a corpus."""

    on: str = "source-change"
    every_hours: float | None = None  # used only when ``on == "interval"``

    def __post_init__(self):
        if self.on not in REINDEX_TRIGGERS:
            raise ValueError(
                f"unknown reindex trigger {self.on!r}; expected {REINDEX_TRIGGERS}"
            )

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReindexPolicy":
        d = d or {}
        return cls(
            on=d.get("on", "source-change"),
            every_hours=d.get("every_hours"),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"on": self.on}
        if self.every_hours is not None:
            out["every_hours"] = self.every_hours
        return out


@dataclass(frozen=True)
class SynopsisPolicy:
    """Whether/when to attach (expensive, LLM-generated) synopses.

    ``downtime_hours`` is a ``[start, end)`` pair of local-clock hours
    (wrapping past midnight is allowed, e.g. ``(22, 6)``); ``None`` means
    "any time". ``scope="recent"`` limits synthesis to artifacts whose
    timestamp is within ``window_days`` (corpora that expose a time signal);
    ``scope="all"`` synthesizes every artifact (bounded by incrementality).
    """

    enabled: bool = False
    scope: str = "recent"  # "recent" | "all"
    window_days: int = 30
    downtime_hours: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "SynopsisPolicy":
        d = d or {}
        dh = d.get("downtime_hours")
        return cls(
            enabled=bool(d.get("enabled", False)),
            scope=d.get("scope", "recent"),
            window_days=int(d.get("window_days", 30)),
            downtime_hours=(int(dh[0]), int(dh[1])) if dh else None,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "scope": self.scope,
            "window_days": self.window_days,
        }
        if self.downtime_hours is not None:
            out["downtime_hours"] = list(self.downtime_hours)
        return out


@dataclass(frozen=True)
class MaintenancePolicy:
    """The background-work policy for one corpus (reindex + synopsis)."""

    reindex: ReindexPolicy = field(default_factory=ReindexPolicy)
    synopsis: SynopsisPolicy = field(default_factory=SynopsisPolicy)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MaintenancePolicy":
        d = d or {}
        return cls(
            reindex=ReindexPolicy.from_dict(d.get("reindex")),
            synopsis=SynopsisPolicy.from_dict(d.get("synopsis")),
        )

    def to_dict(self) -> dict:
        return {"reindex": self.reindex.to_dict(), "synopsis": self.synopsis.to_dict()}

    def merged(self, override: dict | None) -> "MaintenancePolicy":
        """Layer an ``override`` dict on top of this policy (entry over defaults)."""
        if not override:
            return self
        return replace(
            self,
            reindex=ReindexPolicy.from_dict(
                {**self.reindex.to_dict(), **(override.get("reindex") or {})}
            ),
            synopsis=SynopsisPolicy.from_dict(
                {**self.synopsis.to_dict(), **(override.get("synopsis") or {})}
            ),
        )


# --------------------------------------------------------------------------- #
# Smart defaults per kind — the "rules that lead to defaults" for new corpora
# --------------------------------------------------------------------------- #

#: The global fallback policy (any kind without a specific default).
GLOBAL_DEFAULT = MaintenancePolicy()

#: Per-kind maintenance defaults. The rule of thumb encoded here:
#: *small, fully-enumerable* corpora (skills/packages/reports/files) rebuild on
#: source change (cheap, exact); *large, append-mostly, time-stamped* corpora
#: (sessions) rebuild on an interval and keep synopsis off by default, with a
#: downtime window ready for when it is turned on. New kinds register here.
DEFAULTS_BY_KIND: dict[str, dict] = {
    "skills": {"reindex": {"on": "source-change"}},
    "packages": {"reindex": {"on": "source-change"}},
    "reports": {"reindex": {"on": "source-change"}},
    "files": {"reindex": {"on": "source-change"}},
    "sessions": {
        "reindex": {"on": "interval", "every_hours": 24},
        "synopsis": {
            "enabled": False,
            "scope": "recent",
            "window_days": 30,
            "downtime_hours": [2, 6],
        },
    },
}


def default_policy_for_kind(kind: str) -> MaintenancePolicy:
    """The smart-default :class:`MaintenancePolicy` for a corpus ``kind``."""
    return GLOBAL_DEFAULT.merged(DEFAULTS_BY_KIND.get(kind))


def resolve_policy(entry: dict | None) -> MaintenancePolicy:
    """The effective policy for a registry ``entry``: entry over kind over global.

    A v1 entry (no ``maintenance`` key) resolves to its kind's smart default, so
    existing corpora gain a sensible policy without a migration.
    """
    entry = entry or {}
    base = default_policy_for_kind(entry.get("kind", ""))
    return base.merged(entry.get("maintenance"))


def resolve_storage(entry: dict | None) -> dict:
    """The effective storage spec for an ``entry`` (default ``{"backend": "local"}``)."""
    storage = dict((entry or {}).get("storage") or {})
    backend = storage.get("backend", "local")
    if backend not in STORAGE_BACKENDS:
        raise NotImplementedError(
            f"storage backend {backend!r} is not available yet (only "
            f"{STORAGE_BACKENDS}); a vd.Collection backend is tracked in issue #28."
        )
    storage["backend"] = backend
    return storage


# --------------------------------------------------------------------------- #
# Timing predicates (pure; ``now`` is injected for testability)
# --------------------------------------------------------------------------- #


def is_reindex_due(
    policy: MaintenancePolicy,
    last_maintained: datetime | None,
    now: datetime,
) -> bool:
    """Whether a reindex is due under ``policy`` given the last-maintained time.

    - ``source-change`` is always due (the build is incremental and a no-op when
      nothing changed, so running it cannot do harm).
    - ``interval`` is due when never maintained or older than ``every_hours``.
    - ``manual`` is never automatically due.
    """
    on = policy.reindex.on
    if on == "source-change":
        return True
    if on == "manual":
        return False
    # interval
    every = policy.reindex.every_hours
    if not every or last_maintained is None:
        return True
    return now - last_maintained >= timedelta(hours=every)


def in_downtime(policy: MaintenancePolicy, now: datetime) -> bool:
    """Whether ``now`` falls inside the synopsis ``downtime_hours`` window.

    ``None`` window means "any time". A window whose start hour is greater than
    its end hour wraps past midnight (e.g. ``(22, 6)`` is 22:00–06:00).
    """
    dh = policy.synopsis.downtime_hours
    if not dh:
        return True
    start, end = dh
    hour = now.hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight
