"""LLM-backed generation of evaluation cases for :mod:`ir.eval`.

This is the **build-time** companion to the offline scoring harness: it turns a
corpus into a set of :class:`~ir.eval.DiscoveryCase`\\ s by *back-translation* —
given a capability's description, ask an LLM for the user intents that should
route to it. The artifact's id is the free ground-truth label.

Two ideas make the generated set honest:

- **Name masking.** The artifact's *name* is stripped from the description
  *before* it is shown to the generator (and any output that still leaks the
  name is dropped). Otherwise a lexical retriever (the BM25 leg of hybrid) would
  trivially match query→gold on surface name overlap and inflate scores. Many
  real descriptions contain their own name, so masking the **input** — not just
  filtering the output — is what matters.
- **An abstention slice.** A fraction of cases are "no artifact applies" intents
  (empty ``gold``), so the eval can measure correct refusal, not just hits.

The LLM is **injected** (`query_generator` / `abstention_generator` callables),
so the generation *logic* — masking, gold assignment, the leakage guard, the
abstention fraction — is fully testable with a deterministic stub and no network.
The default generators are built lazily on :mod:`oa` (`oa.prompt_function`), so
``import ir.eval_gen`` stays cheap and offline; ``oa`` is only imported when you
actually generate with the real LLM.

The output is plain :class:`~ir.eval.DiscoveryCase` data — freeze it with
:func:`ir.eval.save_cases` (stamping :func:`corpus_signature` into the
``__meta__`` header) and score it with :mod:`ir.eval`. Generation needs a model;
scoring never does.

Quick start::

    import ir
    from ir import eval_gen as eg

    source = ir.CorpusSource.from_skills()
    cases = eg.build_eval_set(source, k=5, corpus_name="skills")   # uses oa
    from ir.eval import save_cases
    save_cases(cases, "skills_eval.jsonl",
               meta={"corpus": "skills", "corpus_signature": eg.corpus_signature(source)})
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .eval import DiscoveryCase

#: Queries generated per artifact by default.
DFLT_QUERIES_PER_ARTIFACT = 5

#: Target share of the case set that is abstention ("no artifact applies").
DFLT_ABSTENTION_FRAC = 0.15

#: Minimum description length (chars) for an artifact to be back-translated.
DFLT_MIN_DESCRIPTION_CHARS = 20

#: What a masked name is replaced with in a description / query.
NAME_PLACEHOLDER = "this capability"

#: Default theme used when generating abstention ("out of scope") intents.
DFLT_ABSTENTION_THEME = "software developer tools"

#: A query generator: ``(description, *, n) -> list[str]`` (n candidate intents).
QueryGenerator = Callable[..., Sequence[str]]

#: An abstention generator: ``(*, n, theme) -> list[str]`` (n out-of-scope intents).
AbstentionGenerator = Callable[..., Sequence[str]]

BACKTRANSLATION_PROMPT = """\
You are generating evaluation data for a tool-retrieval system.

Below is a description of a capability (its name has been hidden on purpose):

{description}

Write {n} natural, varied user requests that this capability should handle.
Rules:
- Do NOT mention any tool, function, skill, or package name.
- Vary phrasing, specificity, and the implied (not explicit) parameters.
- One request per line. No numbering, no quotes, no extra commentary.
"""

ABSTENTION_PROMPT = """\
You are generating "no applicable tool" cases for a tool-retrieval evaluation.

The tool catalog is about: {theme}.

Write {n} natural, plausible user requests that such a catalog should NOT be able
to satisfy because they fall outside its scope.
One request per line. No numbering, no quotes, no extra commentary.
"""


# =========================================================================== #
# Name masking (label-leakage prevention)
# =========================================================================== #


#: Fallback placeholder used when the primary one would itself match the name.
_ALT_PLACEHOLDER = "the tool"


def _ordered_tokens(name: str) -> list[str]:
    """The name's word tokens (split on whitespace/hyphen/underscore), lowercased.

    Tokens shorter than two characters are dropped so a degenerate name (e.g.
    ``"a-"`` or ``"-x"``) cannot collapse to a one-letter pattern that masks
    articles or stray letters throughout the text.
    """
    return [t for t in re.split(r"[\s_-]+", name.lower()) if len(t) >= 2]


def _name_pattern(name: str) -> re.Pattern[str] | None:
    """Whole-word regex matching ``name`` as a contiguous phrase.

    Tolerant of the separator *between* a multi-token name's tokens (any run of
    whitespace / hyphen / underscore), plus the concatenated form — so
    ``"ci-advisor"``, ``"ci advisor"``, ``"ci_advisor"`` and ``"ciadvisor"`` all
    match. Returns ``None`` when nothing maskable survives (see
    :func:`_ordered_tokens`).
    """
    tokens = _ordered_tokens(name)
    if not tokens:
        return None
    phrase = r"[\s_-]+".join(re.escape(t) for t in tokens)
    concat = "".join(re.escape(t) for t in tokens)
    alts = "|".join(sorted({phrase, concat}, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def mask_name(text: str, name: str, *, placeholder: str = NAME_PLACEHOLDER) -> str:
    """Replace occurrences of ``name`` in ``text`` with ``placeholder``.

    Matches the name as a contiguous phrase tolerant of the separator between its
    tokens (``"ci-advisor"`` / ``"ci advisor"`` / ``"ci_advisor"`` all match),
    case-insensitively and whole-word (a short token like ``"ci"`` does not blast
    through ``"specific"``). If ``placeholder`` would itself match the name, a
    neutral alternate is used so the masked text is not self-referential. Used to
    scrub the artifact name out of a description *before* it is generated from.
    """
    pattern = _name_pattern(name)
    if pattern is None:
        return text
    if pattern.search(placeholder):
        placeholder = _ALT_PLACEHOLDER
    return pattern.sub(placeholder, text)


def _leaks_name(text: str, name: str) -> bool:
    """Whether ``text`` reuses the gold ``name`` — matching how BM25 would.

    The lexical retriever is bag-of-words (order-insensitive), so two forms count
    as a leak: (1) the name as a contiguous phrase (any separator), or (2) for a
    multi-token name, *all* its distinctive tokens appearing anywhere (reordered
    or separated). A single shared content word is **not** a leak — that is the
    legitimate semantic overlap the eval is meant to measure.
    """
    pattern = _name_pattern(name)
    if pattern is None:
        return False
    if pattern.search(text):
        return True
    tokens = set(_ordered_tokens(name))
    if len(tokens) >= 2:
        return tokens <= set(re.findall(r"\w+", text.lower()))
    return False


# =========================================================================== #
# Default (oa-backed) generators — lazily built, only when actually used
# =========================================================================== #


#: A genuine leading list marker: a bullet or an ordinal *followed by whitespace*
#: (``- ``, ``* ``, ``• ``, ``1. ``, ``2) ``). Anchored so it never eats a real
#: leading token like ``3D``, ``-9 degrees``, ``.env`` or ``24/7``.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _parse_lines(text: Any) -> list[str]:
    """Parse an LLM list response into clean, non-empty lines.

    Strips only a genuine leading list marker (bullet or ordinal followed by
    whitespace) and surrounding quotes — never an arbitrary leading character —
    so a query like ``"3D modeling help"`` or ``"-9 degrees, what to wear?"``
    keeps its first token intact.
    """
    lines = []
    for raw in str(text).splitlines():
        cleaned = _LIST_MARKER.sub("", raw.strip()).strip().strip("\"'")
        if cleaned:
            lines.append(cleaned)
    return lines


def make_oa_query_generator(
    *, prompt: str = BACKTRANSLATION_PROMPT, **prompt_function_kwargs: Any
) -> QueryGenerator:
    """Build the default back-translation generator on :mod:`oa` (lazy import)."""
    import oa

    fn = oa.prompt_function(
        prompt, egress=_parse_lines, name="backtranslate", **prompt_function_kwargs
    )

    def generate(description: str, *, n: int) -> list[str]:
        return list(fn(description=description, n=n))[:n]

    return generate


def make_oa_abstention_generator(
    *, prompt: str = ABSTENTION_PROMPT, **prompt_function_kwargs: Any
) -> AbstentionGenerator:
    """Build the default abstention generator on :mod:`oa` (lazy import)."""
    import oa

    fn = oa.prompt_function(
        prompt, egress=_parse_lines, name="abstention", **prompt_function_kwargs
    )

    def generate(*, n: int, theme: str) -> list[str]:
        return list(fn(theme=theme, n=n))[:n]

    return generate


# =========================================================================== #
# Case generation
# =========================================================================== #


def _default_describe(raw: Any) -> str:
    """Best-effort describable text from a raw artifact payload."""
    if isinstance(raw, Mapping):
        for key in ("description", "text"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return "\n".join(str(v) for v in raw.values() if isinstance(v, str))
    return str(raw)


def _name_of(artifact_id: str, raw: Any) -> str:
    """The artifact's display name — a non-blank string ``name`` field, else the id.

    A missing, blank, or non-string ``name`` is treated like absent (the id is
    used), so masking and the leakage guard never operate on the literal
    ``"None"`` or an empty string.
    """
    if isinstance(raw, Mapping):
        name = raw.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return artifact_id


def generate_cases(
    source: Any,
    *,
    k: int = DFLT_QUERIES_PER_ARTIFACT,
    mask_names: bool = True,
    query_generator: QueryGenerator | None = None,
    describe: Callable[[Any], str] | None = None,
    min_chars: int = DFLT_MIN_DESCRIPTION_CHARS,
    max_artifacts: int | None = None,
    corpus_name: str | None = None,
) -> list[DiscoveryCase]:
    """Back-translate a corpus source into gold-bearing :class:`DiscoveryCase`\\ s.

    For each artifact in ``source.scope`` (id → raw), extract a description,
    mask the artifact's name out of it, ask ``query_generator`` for ``k`` user
    intents, and emit one case per surviving intent (gold = the artifact id).
    Intents that still leak the name are dropped; artifacts whose description is
    shorter than ``min_chars`` are skipped (and the count is warned, never
    silently dropped).

    Args:
        source: a :class:`~ir.sources.CorpusSource` (anything with ``.items()``).
        k: intents to request per artifact.
        mask_names: scrub the artifact name from the description before
            generating, and drop any generated intent that still contains it.
        query_generator: ``(description, *, n) -> [intent, …]``. Defaults to the
            :mod:`oa`-backed back-translator (built lazily; needs a model).
        describe: ``raw -> description`` (default: the ``description`` / ``text``
            field, else the joined string fields).
        min_chars: skip artifacts whose description is shorter than this.
        max_artifacts: cap how many artifacts to process (for a quick/cheap run);
            when set, artifacts are taken in sorted-id order so the subset is
            deterministic even for filesystem-ordered (``dol``-backed) scopes.
        corpus_name: stamped on each case's ``corpus`` field.

    Returns:
        the generated gold cases.

    Raises:
        ValueError: if ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}.")
    gen = query_generator or make_oa_query_generator()
    describe = describe or _default_describe
    cases: list[DiscoveryCase] = []
    skipped = 0

    items = list(source.items())
    if max_artifacts is not None:
        # Deterministic subset: sort by id so a capped run is reproducible across
        # machines even when the scope iterates in filesystem order.
        items = sorted(items, key=lambda kv: kv[0])[:max_artifacts]

    for artifact_id, raw in items:
        description = describe(raw)
        if not description or len(description.strip()) < min_chars:
            skipped += 1
            continue
        name = _name_of(artifact_id, raw)
        prompt_text = mask_name(description, name) if mask_names else description
        try:
            intents = gen(prompt_text, n=k)
        except Exception as exc:  # a single artifact's generation failing is non-fatal
            warnings.warn(
                f"query generation failed for {artifact_id!r}: {exc}", stacklevel=2
            )
            skipped += 1
            continue
        for intent in intents:
            intent = (intent or "").strip()
            if not intent:
                continue
            if mask_names and _leaks_name(intent, name):
                continue  # output guard: never let the gold name leak into a query
            cases.append(
                DiscoveryCase(
                    query=intent,
                    gold=(artifact_id,),
                    corpus=corpus_name,
                    source_id=artifact_id,
                    metadata={
                        "generator": "backtranslation",
                        "masked": bool(mask_names),
                    },
                )
            )

    if skipped:
        warnings.warn(
            f"generate_cases skipped {skipped} artifact(s) "
            f"(description shorter than {min_chars} chars, or a generation error).",
            stacklevel=2,
        )
    return cases


def generate_abstention_cases(
    n: int,
    *,
    generator: AbstentionGenerator | None = None,
    theme: str = DFLT_ABSTENTION_THEME,
    corpus_name: str | None = None,
) -> list[DiscoveryCase]:
    """Generate ``n`` abstention cases — out-of-scope intents (empty ``gold``)."""
    if n <= 0:
        return []
    gen = generator or make_oa_abstention_generator()
    intents = gen(n=n, theme=theme)
    cases = [
        DiscoveryCase(
            query=intent.strip(),
            gold=(),
            corpus=corpus_name,
            metadata={"generator": "abstention"},
        )
        for intent in intents
        if intent and intent.strip()
    ]
    return cases[:n]


def build_eval_set(
    source: Any,
    *,
    k: int = DFLT_QUERIES_PER_ARTIFACT,
    abstention_frac: float = DFLT_ABSTENTION_FRAC,
    query_generator: QueryGenerator | None = None,
    abstention_generator: AbstentionGenerator | None = None,
    theme: str = DFLT_ABSTENTION_THEME,
    corpus_name: str | None = None,
    **gen_kwargs: Any,
) -> list[DiscoveryCase]:
    """Generate a full eval set — gold cases plus an abstention slice.

    The abstention count is chosen so abstention cases make up (at least)
    ``abstention_frac`` of the returned set: ``ceil(frac * G / (1 - frac))`` for
    ``G`` gold cases. Extra ``gen_kwargs`` flow to :func:`generate_cases`
    (``mask_names``, ``min_chars``, ``max_artifacts``, ``describe``).

    Raises:
        ValueError: if ``abstention_frac`` is outside ``[0, 1)`` (``frac=0``
            means no abstention slice) or ``k`` is less than 1.
    """
    if not 0.0 <= abstention_frac < 1.0:
        raise ValueError(f"abstention_frac must be in [0, 1), got {abstention_frac!r}.")
    gold_cases = generate_cases(
        source,
        k=k,
        query_generator=query_generator,
        corpus_name=corpus_name,
        **gen_kwargs,
    )
    n_abstain = 0
    if gold_cases and 0.0 < abstention_frac < 1.0:
        n_abstain = math.ceil(
            abstention_frac * len(gold_cases) / (1.0 - abstention_frac)
        )
    abstain_cases = generate_abstention_cases(
        n_abstain,
        generator=abstention_generator,
        theme=theme,
        corpus_name=corpus_name,
    )
    return gold_cases + abstain_cases


# =========================================================================== #
# Reproducibility anchor
# =========================================================================== #


def _artifact_ids(source_or_corpus: Any) -> list[str]:
    """Artifact ids of a CorpusSource (its ``scope``) or a built Corpus."""
    scope = getattr(source_or_corpus, "scope", None)
    if scope is not None:
        return list(scope)
    from .eval import corpus_artifact_ids

    return list(corpus_artifact_ids(source_or_corpus))


def corpus_signature(source_or_corpus: Any) -> str:
    """A short, order-independent hash of a corpus's artifact ids.

    Stamp this into a case file's ``__meta__`` header so a frozen eval set can be
    checked against the (live, machine-specific) corpus it was generated from.
    """
    blob = "\n".join(sorted(_artifact_ids(source_or_corpus))).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
