"""The agent skills ship where they should, with spec-clean frontmatter.

Real skill files live in one non-hidden place each: consumer skills in
``ir/data/skills/`` (shipped with ``pip install``), developer skills in
``skills/`` (repo only). ``.claude/skills/`` holds relative symlinks to them,
which hatchling must not walk, or the real files drop out of the sdist.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "ir" / "data" / "skills"
DEV = ROOT / "skills"
SPEC_KEYS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


def _skill_dirs():
    return sorted(p for base in (SHIPPED, DEV) for p in base.iterdir() if p.is_dir())


def _frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    return match.group(1)


def test_there_is_a_consumer_and_a_dev_skill():
    assert (SHIPPED / "ir-search" / "SKILL.md").is_file()
    assert (DEV / "ir-dev-workflow" / "SKILL.md").is_file()


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_frontmatter_is_spec_clean(skill_dir):
    front = _frontmatter(skill_dir / "SKILL.md")
    top_keys = set(re.findall(r"^([A-Za-z][\w-]*):", front, re.M))
    assert top_keys <= SPEC_KEYS, f"non-spec top-level keys: {top_keys - SPEC_KEYS}"
    assert re.search(rf"^name: {re.escape(skill_dir.name)}$", front, re.M)
    audience = re.search(r"^metadata:\n\s+audience: (\w+)$", front, re.M)
    assert audience, "metadata.audience is missing"
    expected = "users" if skill_dir.parent == SHIPPED else "developers"
    assert audience.group(1) == expected


def test_sdist_keeps_the_real_skill_files():
    sdist = pytest.importorskip("hatchling.builders.sdist")
    builder = sdist.SdistBuilder(str(ROOT))
    paths = {
        f.distribution_path.replace("\\", "/") for f in builder.recurse_included_files()
    }
    assert "ir/data/skills/ir-search/SKILL.md" in paths
    assert "skills/ir-dev-workflow/SKILL.md" in paths
