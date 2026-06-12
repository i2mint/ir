"""Unit tests for indexing strategies (artifact -> filter fields + surfaces)."""

from ir.strategy import Chunked, Package, Skill, WholeText, _split


def test_wholetext_str_and_mapping():
    plan = WholeText().decompose("a", "hello world", {"topic": "x"})
    assert len(plan.surfaces) == 1
    assert plan.surfaces[0].kind == "document"
    assert plan.surfaces[0].text == "hello world"
    assert plan.filter_fields == {"topic": "x"}

    plan2 = WholeText().decompose("b", {"text": "body here", "extra": 1}, {})
    assert plan2.surfaces[0].text == "body here"


def test_wholetext_empty_yields_no_surface():
    plan = WholeText().decompose("a", "   ", {})
    assert plan.surfaces == []


def test_chunked_produces_multiple_chunks():
    text = "\n\n".join(
        f"Paragraph number {i} with some filler words." for i in range(40)
    )
    plan = Chunked(chunk_size=200, overlap=20).decompose("doc", text, {"p": "proj"})
    assert len(plan.surfaces) > 1
    assert all(s.kind == "chunk" for s in plan.surfaces)
    assert plan.surfaces[0].metadata["chunk_index"] == 0
    assert plan.filter_fields["p"] == "proj"


def test_skill_embeds_name_and_description_only():
    raw = {"name": "deploy", "description": "Push apps to the server.", "parent": "tw"}
    plan = Skill().decompose("deploy", raw, {})
    assert len(plan.surfaces) == 1
    assert plan.surfaces[0].kind == "capability"
    assert "deploy" in plan.surfaces[0].text and "server" in plan.surfaces[0].text
    assert plan.filter_fields["name"] == "deploy"
    assert plan.filter_fields["parent"] == "tw"


def test_package_description_plus_readme_chunks_and_filter_fields():
    raw = {
        "name": "vd",
        "description": "Facade over vector databases.",
        "readme": "\n\n".join(f"Section {i} text body." for i in range(30)),
        "owner": "ours",
        "deps": ["dol", "numpy"],
    }
    plan = Package(chunk_size=150, overlap=20).decompose("vd", raw, {})
    kinds = {s.kind for s in plan.surfaces}
    assert "description" in kinds and "readme_chunk" in kinds
    assert plan.filter_fields["owner"] == "ours"
    assert plan.filter_fields["name"] == "vd"
    assert plan.filter_fields["has_readme"] is True
    assert plan.filter_fields["deps"] == ["dol", "numpy"]


def test_split_packs_to_chunk_size_not_per_paragraph():
    # 60 short paragraphs (~25 chars each) ~ 1500 chars total.
    text = "\n\n".join(f"short paragraph numbered {i}" for i in range(60))
    chunks = _split(text, chunk_size=300, overlap=30)
    # Packed: far fewer chunks than paragraphs, each near the target size.
    assert len(chunks) < 15
    assert all(len(c) <= 300 + 30 for c in chunks)


def test_split_hard_splits_oversized_paragraph():
    text = "x" * 1000
    chunks = _split(text, chunk_size=200, overlap=20)
    assert len(chunks) >= 5
    assert all(len(c) <= 200 for c in chunks)


def test_split_never_emits_blank_chunks():
    # A >= chunk_size whitespace run with no blank line stays one "paragraph"
    # (the regex only splits on blank lines); hard-splitting it must skip the
    # whitespace-only slices.
    chunks = _split("start" + " " * 500 + "end", chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunked_whitespace_run_yields_no_blank_surfaces():
    text = "start" + " " * 500 + "end"
    plan = Chunked(chunk_size=120, overlap=20).decompose("w", text)
    assert plan.surfaces and all(s.text.strip() for s in plan.surfaces)
    n = len(plan.surfaces)
    # chunk_index contiguous and n_chunks counts only real (stored) chunks.
    assert [s.metadata["chunk_index"] for s in plan.surfaces] == list(range(n))
    assert all(s.metadata["n_chunks"] == n for s in plan.surfaces)
