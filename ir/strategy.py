"""Indexing strategies — the "what do we index?" seam.

An :class:`IndexingStrategy` decomposes one artifact into an
:class:`~ir.base.IndexPlan`: the ``filter_fields`` (hard-filterable metadata,
*not* embedded) and a list of :class:`~ir.base.Surface` (embeddable units). This
is the central extensibility point of ``ir``: a naive corpus uses
:class:`WholeText`; a structured corpus (a package) decomposes into several
heterogeneous surfaces so a query can match the *right part* of an artifact,
and constrains candidates by metadata *before* semantic ranking.

Shipped strategies:

- :class:`WholeText` — one surface = the whole text. The out-of-the-box default.
- :class:`Chunked` — split the text into overlapping chunks (one surface each).
- :class:`Skill` — embed ``name + description`` only (the body stays on disk,
  per the capability-discovery research); name/parent become filter fields.
- :class:`Fields` — a record's named text fields (head + body, balanced windows,
  optional normalizer); one surface per record or per field.
- :class:`Package` — ``name + description`` plus README chunks as surfaces;
  name/owner/deps become filter fields (AI synopsis / problem-class surfaces
  are a documented extension).

Every strategy is a plain callable-ish object with a ``decompose`` method, so
custom strategies need only match the :class:`IndexingStrategy` protocol.
"""

from __future__ import annotations

import functools
import inspect
import math
import re
import unicodedata
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Protocol, runtime_checkable

from .base import IndexPlan, Surface

#: Content-token budget for token-aware chunking under the default embedder.
#: ``all-MiniLM-L6-v2`` caps input at 256 tokens and silently truncates beyond
#: it. We target 250 (not 254) to leave headroom for the two special tokens
#: (``[CLS]``/``[SEP]``) *and* the ±1–2 token drift when an offset-sliced
#: substring is re-tokenized on its own — so an embedded chunk stays within
#: budget and its tail is never silently truncated (see :func:`_chunk_text`).
DFLT_CHUNK_MAX_TOKENS = 250


@runtime_checkable
class IndexingStrategy(Protocol):
    """Decompose one artifact into filter fields + embeddable surfaces."""

    def decompose(
        self, artifact_id: str, raw: Any, metadata: Mapping[str, Any] | None = None
    ) -> IndexPlan: ...


def text_of(raw: Any, text_key: str | None = None) -> str:
    """Best-effort text extraction from a raw artifact payload.

    The SSOT for turning an opaque ``raw`` (a ``str``, a ``Mapping`` with a
    ``text`` field or a ``text_key``, or anything else) into embeddable text —
    reused by the shipped strategies *and* by :func:`ir.synopsis.make_llm_synthesizer`
    so an injected-free synopsis summarizes the same text a strategy would index.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        if text_key is not None:
            return str(raw.get(text_key, "") or "")
        if "text" in raw:
            return str(raw.get("text", "") or "")
        # join string-valued fields as a fallback
        return "\n".join(str(v) for v in raw.values() if isinstance(v, str))
    return str(raw)


#: Backward-compatible private alias (the helper was module-private before it
#: became a cross-module SSOT). Internal call sites may use either name.
_text_of = text_of


def _split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-packing chunker: greedily fill ~``chunk_size`` chunks.

    Splits on blank lines, then packs whole paragraphs up to ``chunk_size``
    (carrying an ``overlap`` tail between chunks). Paragraphs longer than
    ``chunk_size`` are hard-split. This packs to a target size rather than
    emitting one chunk per paragraph.

    Never emits blank chunks: hard-splitting a paragraph that contains a long
    whitespace run (no blank line, so the paragraph regex doesn't split it)
    can slice out whitespace-only pieces — those are skipped, so every chunk
    is embeddable and per-chunk metadata (``chunk_index`` / ``n_chunks``)
    counts only real chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(para) > chunk_size:  # hard-split an oversized paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, chunk_size - overlap)
            chunks.extend(
                piece
                for i in range(0, len(para), step)
                if (piece := para[i : i + chunk_size]).strip()
            )
            continue
        if cur and len(cur) + 2 + len(para) > chunk_size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail}\n\n{para}".strip() if tail else para
        else:
            cur = f"{cur}\n\n{para}".strip() if cur else para
    if cur:
        chunks.append(cur)
    return chunks


@functools.lru_cache(maxsize=8)
def _tokenizer_for(model_name: str):
    """A cached *fast* tokenizer for *model_name*, or ``None`` if unavailable.

    Lazy (keeps ``import ir`` offline) and self-degrading: any import/load
    failure — or a slow tokenizer that cannot report char offsets — returns
    ``None`` so token-aware chunking falls back to char-based splitting, mirroring
    the embedder's degrade-to-hashing discipline (:mod:`ir.embed`).
    """
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    # A bare sentence-transformers name (e.g. "all-MiniLM-L6-v2") is a valid
    # SentenceTransformer id but not a Hub repo id — AutoTokenizer needs the
    # "sentence-transformers/" prefix. Try the name as given, then prefixed.
    candidates = [model_name]
    if "/" not in model_name:
        candidates.append(f"sentence-transformers/{model_name}")
    for candidate in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(candidate)
        except Exception:
            continue
        # Offset mapping (used to cut the original text at token boundaries) needs
        # a Rust "fast" tokenizer; a slow one would force a lossy id round-trip.
        if getattr(tok, "is_fast", False):
            return tok
    return None


def _split_by_tokens(
    text: str, *, tokenizer: Any, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split *text* into windows of at most *max_tokens* tokens, text preserved.

    Cuts the **original** string at token boundaries using the tokenizer's char
    offset-mapping — so each stored surface stays verbatim (for lexical search and
    disclosure), never a lossy decode of token ids — with *overlap_tokens* of
    overlap between consecutive windows.
    """
    text = text.strip()
    if not text:
        return []
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n <= max_tokens:
        return [text]
    step = max(1, max_tokens - overlap_tokens)
    chunks: list[str] = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        piece = text[offsets[i][0] : offsets[j - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if j >= n:
            break
        i += step
    return chunks


def _chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_tokens: int | None = None,
    token_model: str | None = None,
) -> list[str]:
    """Chunk *text* by tokens when *max_tokens* is set, else by characters.

    Token mode bounds every chunk to the embedder's token budget, so no chunk is
    silently truncated at embed time; the token overlap is derived from the char
    ``overlap / chunk_size`` ratio. It degrades to char-based splitting (with a
    warning) when the tokenizer for *token_model* cannot be loaded. Char mode
    (``max_tokens=None``) is the historical behavior.
    """
    if not max_tokens:
        return _split(text, chunk_size=chunk_size, overlap=overlap)
    from .embed import DEFAULT_MODEL

    model = token_model or DEFAULT_MODEL
    tokenizer = _tokenizer_for(model)
    if tokenizer is None:
        warnings.warn(
            f"token-aware chunking requested (max_tokens={max_tokens}) but no fast "
            f"tokenizer for {model!r} is available; falling back to char-based "
            f"chunking (chunks may exceed the embedder's token budget and be "
            f"truncated). Install `transformers` for token-aware chunking.",
            stacklevel=2,
        )
        return _split(text, chunk_size=chunk_size, overlap=overlap)
    overlap_tokens = round(max_tokens * overlap / chunk_size) if chunk_size else 0
    return _split_by_tokens(
        text, tokenizer=tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )


class WholeText:
    """One surface = the entire text. Sensible default for a naive corpus."""

    def __init__(self, *, text_key: str | None = None, kind: str = "document"):
        self.text_key = text_key
        self.kind = kind

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        text = _text_of(raw, self.text_key)
        surfaces = (
            [Surface(artifact_id, self.kind, text, granularity="document")]
            if text.strip()
            else []
        )
        return IndexPlan(filter_fields=meta, surfaces=surfaces)


def _nonblank(chunks: Iterable[str]) -> list[str]:
    """Drop whitespace-only chunks.

    The contract every windowing strategy keeps (:class:`Chunked`,
    :class:`Fields`): a splitter may cut a long whitespace run into windows that
    strip to nothing, and those must never become surfaces (nor count in
    ``chunk_index`` / ``n_chunks``).

    >>> _nonblank(["abc", "   ", "", "def"])
    ['abc', 'def']
    """
    return [c for c in chunks if c.strip()]


class Chunked:
    """Split the artifact's text into overlapping chunk surfaces.

    By default chunks are packed to ~``chunk_size`` **characters**. Set
    ``max_tokens`` to chunk by **tokens** instead — each chunk is bounded to
    ``max_tokens`` tokens of ``token_model`` (default: the default embedding
    model), so no chunk overflows the embedder's sequence limit and is silently
    truncated. Token mode degrades to char mode with a warning if the tokenizer
    is unavailable. The char ``chunk_size`` / ``overlap`` still set the overlap
    *ratio* used in token mode.

    ``split`` swaps the character splitter for a ``(text, *, size, overlap) ->
    list[str]`` callable or ``"module:attr"`` reference, e.g.
    :func:`balanced_windows` (equal-sized windows, no short tail) or
    :func:`word_windows` (word-aligned cuts), with ``size=chunk_size``. It is
    exclusive with ``max_tokens``. ``split_id`` names a lambda/closure for
    re-indexing, as for :class:`Fields`.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1200,
        overlap: int = 200,
        text_key: str | None = None,
        kind: str = "chunk",
        max_tokens: int | None = None,
        token_model: str | None = None,
        split: Callable[..., list[str]] | str | None = None,
        split_id: str | None = None,
    ):
        if split is not None and max_tokens is not None:
            raise ValueError("split= and max_tokens= are exclusive: pick one splitter")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_key = text_key
        self.kind = kind
        self.max_tokens = max_tokens
        self.token_model = token_model
        self.split = split
        self.split_id = _normalizer_id(split, split_id, what="split")

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        text = _text_of(raw, self.text_key)
        if self.split is None:
            chunks = _chunk_text(
                text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                max_tokens=self.max_tokens,
                token_model=self.token_model,
            )
        else:
            from .sources import resolve_fetcher

            splitter = (
                resolve_fetcher(self.split)
                if isinstance(self.split, str)
                else self.split
            )
            chunks = _nonblank(
                splitter(text, size=self.chunk_size, overlap=self.overlap)
            )
        surfaces = [
            Surface(
                artifact_id,
                self.kind,
                chunk,
                granularity="chunk",
                metadata={"chunk_index": i, "n_chunks": len(chunks)},
            )
            for i, chunk in enumerate(chunks)
        ]
        return IndexPlan(filter_fields=meta, surfaces=surfaces)


class Skill:
    """Capability strategy: embed ``name + description`` only.

    The body (SKILL.md) is loaded post-selection and is *not* indexed; name and
    parent are filter fields.
    """

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        raw = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        name = str(raw.get("name", artifact_id))
        description = str(raw.get("description", "") or "")
        filter_fields = {
            "name": name,
            "parent": raw.get("parent"),
            **meta,
        }
        text = f"{name}\n\n{description}".strip()
        surfaces = [Surface(artifact_id, "capability", text, granularity="field")]
        return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)


def is_simple_param(value: Any) -> bool:
    """Whether a constructor parameter is JSON-simple (scalar, or a tuple/list of scalars).

    The shared test behind a strategy's persisted spec (:func:`strategy_to_spec`)
    and its build identity (``ir.index._strategy_id``), so a parameter such as
    :class:`Fields`'s ``fields`` both round-trips through the registry and
    re-decomposes a corpus when it changes.
    """
    scalar = (str, int, float, bool, type(None))
    if isinstance(value, (tuple, list)):
        return all(isinstance(v, scalar) for v in value)
    return isinstance(value, scalar)


#: Latin letters with no Unicode decomposition (strokes, ligature-letters) that a
#: search fold should still map to ASCII, as ICU's Latin-ASCII transform does.
_LATIN_LETTER_FOLDS = str.maketrans(
    {
        "ø": "o",
        "đ": "d",
        "ð": "d",
        "ł": "l",
        "ħ": "h",
        "ŧ": "t",
        "ı": "i",
        "æ": "ae",
        "œ": "oe",
        "þ": "th",
    }
)


@functools.lru_cache(maxsize=4096)
def _fold_char(ch: str) -> tuple[str, bool]:
    """``(folded, is_latin_or_greek)`` for one casefolded character (cached)."""
    if not unicodedata.name(ch, "").startswith(("LATIN", "GREEK")):
        return ch, False
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c)
    )
    return stripped.translate(_LATIN_LETTER_FOLDS), True


def fold_text(text: str) -> str:
    """Case-fold *text* and strip accents from Latin and Greek letters only.

    Applies NFKC first (full-width letters, ligatures and compatibility forms fold
    in), then ``casefold``, then removes accents of Latin/Greek letters, mapping
    strokes and ligature-letters (``ø đ ł æ œ þ``) to ASCII. Safe for mixed-script
    text: marks that carry meaning elsewhere are left alone (kana voicing ``が``,
    Hangul syllables, Devanagari virama, Thai tone marks, Cyrillic ``й``/``ё``).
    Use it as a ``Fields(normalize=...)`` normalizer (by reference:
    ``"ir.strategy:fold_text"``) **and** apply it to queries, e.g.
    ``formulate=fold_text`` (a ``str -> str`` is a valid formulator), since folding
    acts on whichever side runs it. A synopsis written by an LLM
    (``ir.with_synopsis``) is not folded.

    >>> fold_text("Crème BRÛLÉE Straße Đà Nẵng Łódź")
    'creme brulee strasse da nang lodz'
    >>> fold_text("Ｃｒèｍｅ")
    'creme'
    >>> fold_text("Αθήνα")
    'αθηνα'
    >>> fold_text("がぎ 한국어 हिन्दी Йё ไก่")
    'がぎ 한국어 हिन्दी йё ไก่'
    """
    if text.isascii():
        return text.casefold()
    out: list[str] = []
    after_letter = False  # was the last kept char a Latin/Greek letter?
    for ch in unicodedata.normalize("NFKC", text).casefold():
        if after_letter and unicodedata.combining(ch):
            continue  # a stray accent on a Latin/Greek letter
        folded, after_letter = _fold_char(ch)
        out.append(folded)
    return "".join(out)


def balanced_windows(text: str, *, size: int, overlap: int = 0) -> list[str]:
    """Split *text* into the fewest windows of at most *size* chars, **near-equal in length**.

    ``n = ceil((L - overlap) / (size - overlap))`` windows, each ``ceil((L +
    (n - 1) * overlap) / n)`` long, so a text just past the limit becomes two
    halves, not one full window plus a sliver (which length normalisation in
    BM25 would favour).

    >>> balanced_windows("a" * 10, size=10)
    ['aaaaaaaaaa']
    >>> [len(w) for w in balanced_windows("a" * 11, size=10)]
    [6, 5]
    >>> [len(w) for w in balanced_windows("a" * 25, size=10, overlap=2)]
    [10, 10, 9]
    """
    if not 0 <= overlap < size:
        raise ValueError(
            f"need 0 <= overlap < size, got overlap={overlap}, size={size}"
        )
    text = text.strip()
    length = len(text)
    if length <= size:
        return [text] if text else []
    n = math.ceil((length - overlap) / (size - overlap))
    width = math.ceil((length + (n - 1) * overlap) / n)
    step = width - overlap
    return [text[i * step : i * step + width] for i in range(n)]


def word_windows(text: str, *, size: int, overlap: int = 0) -> list[str]:
    """Like :func:`balanced_windows`, but each cut moves back to the nearest whitespace.

    A window that would end mid-word ends at the last whitespace (a space, a
    newline, a tab) inside it instead (when there is one), and the next window
    starts at the following word, so no word is split. Windows are therefore *at most* ``size`` long but not
    near-equal. ``overlap`` is a character target, rounded to word starts.

    >>> word_windows("alpha beta gamma delta", size=12)
    ['alpha beta', 'gamma delta']
    >>> word_windows("short", size=12)
    ['short']
    """
    if not 0 <= overlap < size:
        raise ValueError(
            f"need 0 <= overlap < size, got overlap={overlap}, size={size}"
        )
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    windows, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = _last_space(text, start + 1, end + 1)
            if cut > start:
                end = cut
        windows.append(text[start:end].strip())
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        if overlap and nxt > start:
            space = _next_space(text, nxt)
            nxt = space + 1 if 0 <= space < end else end
        start = nxt
        while start < n and text[start].isspace():
            start += 1
    return [w for w in windows if w]


_SPACE = re.compile(r"\s")


def _last_space(text: str, lo: int, hi: int) -> int:
    """Index of the last whitespace character in ``text[lo:hi]``, or -1."""
    for i in range(min(hi, len(text)) - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def _next_space(text: str, lo: int) -> int:
    """Index of the first whitespace character at or after ``lo``, or -1."""
    m = _SPACE.search(text, lo)
    return m.start() if m else -1


def _field_text(value: Any) -> str:
    """A record field's text: strings as-is, lists joined, ``None`` empty."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(str(v) for v in value)
    return str(value)


def strategy_normalizer(strategy) -> Callable[[str], str] | str | None:
    """The ``normalize`` an indexing *strategy* applies to its text, as given.

    That is the strategy's ``normalize`` attribute (a callable or a
    ``"module:attr"`` reference), or, for a wrapper such as
    :func:`ir.with_synopsis`, that of the strategy it wraps; ``None`` when the
    indexed text is not normalized. :func:`ir.build` records it so a search can
    fold its query with the same function.

    >>> strategy_normalizer(Fields(fields=["about"], normalize="ir.strategy:fold_text"))
    'ir.strategy:fold_text'
    >>> strategy_normalizer(Fields(fields=["about"])) is None
    True
    """
    normalize = getattr(strategy, "normalize", None)
    if normalize is not None:
        return normalize
    for value in vars(strategy).values() if hasattr(strategy, "__dict__") else ():
        if value is not strategy and hasattr(value, "decompose"):
            found = strategy_normalizer(value)
            if found is not None:
                return found
    return None


def _resolve_normalizer(normalize) -> Callable[[str], str]:
    """A ``str -> str`` callable from a callable, a ``"module:attr"`` ref, or ``None``."""
    if normalize is None:
        return lambda text: text
    from .sources import resolve_fetcher  # the shared "module:attr" resolver

    return resolve_fetcher(normalize)


def _normalizer_id(
    normalize, explicit: str | None, *, what: str = "normalize"
) -> str | None:
    """Identity stamp of a callable normalizer (``None`` for none or a named ref).

    An import reference is its own identity (it is a persisted scalar). An
    unnamed lambda or local closure has none, so the user must pass
    ``normalize_id=`` to have a change re-index the corpus.
    """
    if normalize is None or isinstance(normalize, str):
        if explicit is not None:
            raise ValueError(f"{what}_id requires a callable `{what}`")
        return None
    if explicit:
        return explicit
    qualname = getattr(normalize, "__qualname__", "") or ""
    if qualname and "<lambda>" not in qualname and "<locals>" not in qualname:
        module = getattr(normalize, "__module__", None) or type(normalize).__module__
        return f"{module}.{qualname}"
    warnings.warn(
        f"Fields: could not derive a stable identity for `{what}` (an unnamed "
        f"lambda, a local closure or a partial), so changing it will not re-index; "
        f"pass {what}_id=... to track it, or a 'module:attr' reference.",
        stacklevel=3,
    )
    return "custom"


class Fields:
    """Index a record's named text fields (a catalogue item: name, place, prose...).

    ``fields`` are the body keys, read in order; ``head`` are keys (a name, a
    place) that lead the record. Two layouts:

    - ``layout="record"`` (default): one surface per window of the body, with the
      head text **prefixed to every window**, so a word in the name and a word in
      the place only score together because they share a surface. Measured on 12
      judged requests over ~44k records: BM25 hit@1 0.92 vs 0.58 for ``"fields"``.
    - ``layout="fields"``: one surface per field (head and body), ``kind`` = the
      field name, so a query can target one field with ``surfaces={...}``.

    ``normalize`` (a ``str -> str`` callable, or a ``"module:attr"`` reference to
    one, case/accent folding...) is applied to every text before indexing, and
    :func:`ir.build` records it on the corpus so every search folds its query
    with the same function (see :class:`ir.index.Corpus`). A reference is
    persisted in a registry spec and in the built corpus, and is its own
    identity; a callable is not persisted (a reopened corpus warns that its
    queries are no longer folded, unless ``open_corpus(query_normalize=...)``
    hands it back), and is identified by its qualified name or an explicit
    ``normalize_id`` (required for a lambda or closure).
    ``max_chars`` / ``overlap`` window the *body* with :func:`balanced_windows`
    (``split`` swaps the splitter: a ``(text, *, size, overlap) -> list[str]``
    callable or ``"module:attr"`` reference, e.g. :func:`word_windows` for
    word-aligned cuts; ``split_id`` names a lambda/closure for re-indexing) (the head is prefixed to each window and not itself windowed, so keep it short).
    ``kind`` names the surface in the ``"record"`` layout; ``"fields"`` uses the
    field names. A key listed in both ``head`` and ``fields`` is read once.
    :attr:`reads` lists every key the strategy reads, so a change signal can cover them.

    >>> plan = Fields(fields=["about"], head=["name", "place"]).decompose(
    ...     "r1", {"name": "Louvre", "place": "Paris", "about": "A museum."})
    >>> [(s.kind, s.text) for s in plan.surfaces]
    [('record', 'Louvre\\nParis\\nA museum.')]
    """

    def __init__(
        self,
        fields: Sequence[str],
        *,
        head: Sequence[str] = (),
        layout: str = "record",
        normalize: Callable[[str], str] | str | None = None,
        normalize_id: str | None = None,
        max_chars: int = 1200,
        overlap: int = 200,
        kind: str = "record",
        split: Callable[..., list[str]] | str | None = None,
        split_id: str | None = None,
    ):
        if layout not in ("record", "fields"):
            raise ValueError(f"layout must be 'record' or 'fields', got {layout!r}")
        if isinstance(fields, str) or isinstance(head, str):
            raise TypeError("fields and head are sequences of keys, not a bare string")
        if not (isinstance(max_chars, int) and isinstance(overlap, int)):
            raise TypeError("max_chars and overlap must be integers")
        if not 0 <= overlap < max_chars:
            raise ValueError(
                f"need 0 <= overlap < max_chars, got {overlap}, {max_chars}"
            )
        # dict.fromkeys: drop duplicates, keep order; a head key is not re-read as body
        self.head = tuple(dict.fromkeys(head))
        self.fields = tuple(k for k in dict.fromkeys(fields) if k not in self.head)
        self.layout = layout
        self.normalize = normalize
        self.normalize_id = _normalizer_id(normalize, normalize_id)
        self.max_chars = max_chars
        self.overlap = overlap
        self.kind = kind
        self.split = split
        self.split_id = _normalizer_id(split, split_id, what="split")

    @property
    def reads(self) -> tuple[str, ...]:
        """Every record key this strategy reads (for change signals)."""
        return (*self.head, *self.fields)

    @property
    def synopsis_fields(self) -> tuple[str, ...]:
        """The keys a synopsis should summarize (``ir.with_synopsis`` reads this)."""
        return self.reads

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        record = raw if isinstance(raw, Mapping) else {}
        norm = _resolve_normalizer(self.normalize)
        from .sources import resolve_fetcher

        def text(key):
            return norm(_field_text(record.get(key))).strip()

        splitter = (
            balanced_windows
            if self.split is None
            else (
                resolve_fetcher(self.split)
                if isinstance(self.split, str)
                else self.split
            )
        )

        def windows(t):
            return _nonblank(splitter(t, size=self.max_chars, overlap=self.overlap))

        if self.layout == "fields":
            surfaces = []
            for key in self.reads:
                parts = windows(text(key))
                surfaces += [
                    Surface(
                        artifact_id,
                        key,
                        part,
                        granularity="field",
                        metadata={"chunk_index": i, "n_chunks": len(parts)},
                    )
                    for i, part in enumerate(parts)
                ]
        else:
            head = "\n".join(t for t in map(text, self.head) if t)
            body = windows("\n".join(t for t in map(text, self.fields) if t))
            parts = body or ([""] if head else [])
            surfaces = [
                Surface(
                    artifact_id,
                    self.kind,
                    "\n".join(p for p in (head, part) if p),
                    granularity="chunk" if len(parts) > 1 else "document",
                    metadata={"chunk_index": i, "n_chunks": len(parts)},
                )
                for i, part in enumerate(parts)
            ]
        return IndexPlan(filter_fields=meta, surfaces=surfaces)


#: Shipped strategies addressable by name, for persisting an
#: :class:`IndexingStrategy` in a registry entry and reconstructing it (#58).
#: New shipped strategies register here so a corpus can name its segmentation.
STRATEGY_REGISTRY: dict[str, type] = {
    "WholeText": WholeText,
    "Chunked": Chunked,
    "Skill": Skill,
    "Fields": Fields,
}


def strategy_to_spec(strategy: Any) -> dict:
    """A ``{"name", "params"}`` spec for a shipped, scalar-param strategy.

    Captures only scalar constructor parameters (the same identity surface
    :func:`ir.index._strategy_id` stamps), so it round-trips the shipped
    strategies. A custom strategy, or one wrapping another (e.g. the
    :func:`ir.with_synopsis` wrapper), is **not** captured here — those are set
    programmatically at build time / by the maintenance layer, not persisted as a
    segmentation spec.
    """
    name = type(strategy).__name__
    ctor = inspect.signature(type(strategy)).parameters
    open_ended = any(p.kind is p.VAR_KEYWORD for p in ctor.values())
    params = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in vars(strategy).items()
        if is_simple_param(v) and (open_ended or k in ctor)  # not identity-only stamps
    }
    return {"name": name, "params": params}


def strategy_from_spec(spec: Mapping[str, Any] | None) -> "IndexingStrategy | None":
    """Reconstruct a shipped strategy from a ``{"name", "params"}`` spec.

    ``None`` (no persisted strategy) returns ``None`` so the caller falls back to
    the source preset's default strategy — the back-compatible behavior for v1
    registry entries.
    """
    if not spec:
        return None
    name = spec.get("name")
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown strategy {name!r}; known: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls(**dict(spec.get("params") or {}))


def _default_deps_text(deps: list[str]) -> str:
    """Prefix-form serialization of bare dependency names for the ``deps`` surface.

    Strips version specifiers / extras / markers (via :func:`ir.graph._dep_name`),
    de-duplicates, preserves order, and drops empties — e.g.
    ``["numpy>=1", "sentence-transformers", "numpy"]`` ->
    ``"Depends on: numpy, sentence-transformers"``. Returns ``""`` when there are
    no usable names (``decompose`` then drops the empty surface).
    """
    from .graph import _dep_name

    names = list(dict.fromkeys(n for n in (_dep_name(d) for d in deps) if n))
    return "Depends on: " + ", ".join(names) if names else ""


class Package:
    """Package strategy: ``name + description`` surface plus README chunks.

    Filter fields capture ownership (ours vs third-party), name, deps. AI
    synopsis / problem-class surfaces are a documented extension point.

    With ``embed_deps=True``, ``decompose`` additionally emits one
    ``Surface(kind="deps", granularity="field")`` whose text is a prefix-form
    serialization of the **bare** dependency names (``deps_template``, default
    :func:`_default_deps_text`) — so a query for a domain matches a package by the
    libraries it depends on (e.g. ``sentence-transformers`` -> embeddings,
    ``networkx`` -> graphs), and the BM25 leg picks up exact dep-token matches. The
    deps bag is kept **separate from prose** (its own surface) so a rare library
    name is not diluted, and deps remain a filter field regardless. ``embed_deps``
    defaults ``False`` (today's behavior); it folds into the strategy id, so
    toggling it re-decomposes incrementally. The deps surface is appended **last**,
    leaving the ``description`` (position 0) and ``readme_chunk`` indices unchanged.

    Surface indexing: the ``description`` surface (kept whenever *name* or
    *description* is non-empty) occupies plan position 0, so ``readme_chunk``
    *j* is stored with
    ``Record.surface_index == j + 1`` while its surface metadata says
    ``chunk_index == j`` — ``surface_index`` is plan-global, ``chunk_index``
    per-kind (see :meth:`ir.base.Record.make_id`). Never derive sibling record
    ids from ``chunk_index``; use the ledger
    (:func:`ir.retrieve.records_for_artifact`). ``n_chunks`` is stamped on
    readme chunks at decompose time, but corpora built before the stamp keep
    records without it until the artifact re-indexes (content / embedder /
    strategy change) — read it with ``metadata.get("n_chunks")``.
    """

    #: The record fields a synopsis is written from (see :func:`ir.synopsis.with_synopsis`):
    #: prose only. The record's ``path`` (a local absolute path) and ``deps`` never go to an LLM.
    synopsis_fields = ("name", "description", "readme")

    def __init__(
        self,
        *,
        chunk_size: int = 1500,
        overlap: int = 200,
        embed_deps: bool = False,
        deps_template: Callable[[list[str]], str] | None = None,
        max_tokens: int | None = None,
        token_model: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embed_deps = embed_deps
        self.deps_template = deps_template
        self.max_tokens = max_tokens
        self.token_model = token_model

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        raw = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        name = str(raw.get("name", artifact_id))
        description = str(raw.get("description", "") or "")
        readme = str(raw.get("readme", "") or "")
        filter_fields = {
            "name": name,
            "owner": raw.get("owner"),
            "has_readme": bool(readme.strip()),
            "deps": list(raw.get("deps", []) or []),
            **meta,
        }
        surfaces = [
            Surface(
                artifact_id,
                "description",
                f"{name}: {description}".strip(": ").strip(),
                granularity="field",
            )
        ]
        chunks = _chunk_text(
            readme,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            max_tokens=self.max_tokens,
            token_model=self.token_model,
        )
        for i, chunk in enumerate(chunks):
            surfaces.append(
                Surface(
                    artifact_id,
                    "readme_chunk",
                    chunk,
                    granularity="chunk",
                    metadata={"chunk_index": i, "n_chunks": len(chunks)},
                )
            )
        if self.embed_deps:
            template = self.deps_template or _default_deps_text
            deps_text = template(list(filter_fields["deps"]))
            if deps_text and deps_text.strip():
                # Appended last: keeps the description (0) and readme_chunk indices
                # stable, so the surface_index contract holds for existing corpora.
                surfaces.append(
                    Surface(artifact_id, "deps", deps_text, granularity="field")
                )
        # Drop empty surfaces (e.g. no description and no README).
        surfaces = [s for s in surfaces if s.text.strip()]
        return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)


class ClaudeTurn:
    """Index a Claude Code session turn-pair: user prompt + assistant summary.

    Two surfaces by default — ``user_prompt`` (what the human asked) and
    ``assistant_summary`` (the assistant's *final* end-of-turn text, the
    highest-signal "here's what I did"; the deliberation before it is mostly
    noise) — so a query can target either side via ``surfaces={"user_prompt"}`` /
    ``{"assistant_summary"}``. With ``include_full=True`` a third
    ``assistant_full`` surface (all the turn's assistant natural-language text)
    trades noise for recall — off by default. Session / project / time / model /
    tool-use become hard-filter fields.

    The raw artifact is a turn-pair record (see
    :func:`priv.claude_transcripts.turn_pair_records`): a mapping with
    ``user_prompt`` / ``assistant_summary`` / ``assistant_full`` plus metadata.
    """

    def __init__(self, *, include_full: bool = False):
        self.include_full = include_full

    def decompose(self, artifact_id, raw, metadata=None) -> IndexPlan:
        meta = dict(metadata or {})
        raw = raw if isinstance(raw, Mapping) else {}
        user = str(raw.get("user_prompt", "") or "").strip()
        summary = str(raw.get("assistant_summary", "") or "").strip()
        full = str(raw.get("assistant_full", "") or "").strip()
        title = str(raw.get("session_title", "") or "").strip()
        filter_fields = {
            "session_id": raw.get("session_id"),
            "project": raw.get("project"),
            "cwd": raw.get("cwd"),
            "git_branch": raw.get("git_branch"),
            "timestamp": raw.get("timestamp"),
            "model": raw.get("model"),
            "has_tool_use": bool(raw.get("has_tool_use")),
            "session_title": raw.get("session_title"),
            **meta,
        }
        surfaces = []
        # A dedicated per-session title record indexes just the title (a cheap,
        # searchable "what was this session about" surface).
        if raw.get("record_type") == "session_title":
            if title:
                surfaces.append(
                    Surface(artifact_id, "session_title", title, granularity="field")
                )
            return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)
        if user:
            surfaces.append(
                Surface(artifact_id, "user_prompt", user, granularity="field")
            )
        if summary:
            surfaces.append(
                Surface(artifact_id, "assistant_summary", summary, granularity="field")
            )
        if self.include_full and full and full != summary:
            surfaces.append(
                Surface(artifact_id, "assistant_full", full, granularity="document")
            )
        return IndexPlan(filter_fields=filter_fields, surfaces=surfaces)


# Register the strategies defined after STRATEGY_REGISTRY's literal above.
STRATEGY_REGISTRY["Package"] = Package
STRATEGY_REGISTRY["ClaudeTurn"] = ClaudeTurn
