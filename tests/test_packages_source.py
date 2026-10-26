"""Offline tests for the ``packages`` source's repo enumeration (i2mint/ir#117)."""

import json

import pytest

import ir
from ir import sources


def _repo(root, rel, *, pyproject_desc=None, package_json_desc=None, readme=None):
    d = root / rel
    d.mkdir(parents=True)
    if pyproject_desc is not None:
        (d / "pyproject.toml").write_text(
            f'[project]\nname = "{d.name}"\ndescription = "{pyproject_desc}"\n'
        )
    if package_json_desc is not None:
        (d / "package.json").write_text(json.dumps({"description": package_json_desc}))
    if readme is not None:
        (d / "README.md").write_text(readme)
    return d


def test_repos_seam_indexes_non_python_repos(tmp_path):
    py = _repo(tmp_path, "t/pyrepo", pyproject_desc="A Python package")
    js = _repo(
        tmp_path, "tt/jsapp", package_json_desc="Flashcards with spaced repetition"
    )
    data = _repo(tmp_path, "misc/papers", readme="Past papers and solutions")
    src = ir.CorpusSource.from_packages(repos=lambda: [py, js, data])
    assert src.scope["jsapp"]["description"] == "Flashcards with spaced repetition"
    assert src.scope["papers"]["readme"].startswith("Past papers")


def test_pyproject_description_is_read(tmp_path):
    pytest.importorskip("tomllib" if _has("tomllib") else "tomli")
    py = _repo(tmp_path, "t/pyrepo", pyproject_desc="A Python package")
    src = ir.CorpusSource.from_packages(repos=lambda: [py])
    assert src.scope["pyrepo"]["description"] == "A Python package"


def _has(module):
    import importlib.util

    return importlib.util.find_spec(module) is not None


def test_same_named_dirs_do_not_overwrite(tmp_path):
    a = _repo(tmp_path, "t/thing", package_json_desc="python thing")
    b = _repo(tmp_path, "tt/thing", package_json_desc="js thing")
    scope = ir.CorpusSource.from_packages(repos=lambda: [a, b]).scope
    assert scope["thing"]["description"] == "python thing"
    assert scope["thing@tt"]["description"] == "js thing"


def test_package_repo_dirs_unions_manifest_and_fleet(tmp_path, monkeypatch):
    a = _repo(tmp_path, "t/a", pyproject_desc="a")
    b = _repo(tmp_path, "tt/b", package_json_desc="b")
    manifest = tmp_path / "pkgs.pth"
    manifest.write_text(f"{a}\n")
    monkeypatch.setattr(sources, "_fleet_repo_dirs", lambda: [a, b])
    assert sources.package_repo_dirs(manifest) == [a, b]


def test_package_repo_dirs_without_priv_is_manifest_only(tmp_path, monkeypatch):
    a = _repo(tmp_path, "t/a", pyproject_desc="a")
    manifest = tmp_path / "pkgs.pth"
    manifest.write_text(f"{a}\n")
    monkeypatch.setattr(sources, "_fleet_repo_dirs", lambda: [])
    assert sources.package_repo_dirs(manifest) == [a]


def test_exclude_drops_matching_repos(tmp_path):
    keep = _repo(tmp_path, "q/open", readme="fine")
    drop = _repo(tmp_path, "q/secret-restricted", readme="never read me")
    src = ir.CorpusSource.from_packages(
        repos=lambda: [keep, drop], exclude=["*/q/*-restricted"]
    )
    assert list(src.scope) == ["open"]


def test_exclude_defaults_to_local_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("IR_CONFIG_DIR", str(tmp_path / "cfg"))
    keep = _repo(tmp_path, "t/a", readme="a")
    drop = _repo(tmp_path, "t/b", readme="b")
    from ir.config import exclude_path

    exclude_path().write_text("# private\n*/t/b\n")
    src = ir.CorpusSource.from_packages(repos=lambda: [keep, drop])
    assert list(src.scope) == ["a"]


_SETUP_CFG = """[metadata]
name = old
description = An older package

[options]
install_requires =
    networkx>=3
    # a comment
    dol ; python_version >= "3.10"

"""


def test_setup_cfg_metadata_is_read(tmp_path):
    d = tmp_path / "t" / "old"
    d.mkdir(parents=True)
    (d / "setup.cfg").write_text(_SETUP_CFG)
    (d / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\n')
    rec = ir.CorpusSource.from_packages(repos=lambda: [d]).scope["old"]
    assert rec["description"] == "An older package"
    assert rec["deps"] == ["networkx>=3", 'dol ; python_version >= "3.10"']


def test_pyproject_project_fields_win_over_setup_cfg(tmp_path):
    pytest.importorskip("tomllib" if _has("tomllib") else "tomli")
    d = _repo(tmp_path, "t/both", pyproject_desc="From pyproject")
    with (d / "pyproject.toml").open("a") as f:
        f.write("dependencies = []\n")
    (d / "setup.cfg").write_text(_SETUP_CFG)
    rec = ir.CorpusSource.from_packages(repos=lambda: [d]).scope["both"]
    assert rec["description"] == "From pyproject"
    assert rec["deps"] == []  # declared (empty) in [project]: setup.cfg is not read


def test_unparseable_setup_cfg_warns_and_is_ignored(tmp_path):
    d = tmp_path / "t" / "bad"
    d.mkdir(parents=True)
    (d / "setup.cfg").write_text("no section header\n")
    with pytest.warns(UserWarning, match="setup.cfg"):
        rec = ir.CorpusSource.from_packages(repos=lambda: [d]).scope["bad"]
    assert rec["deps"] == [] and rec["description"] == ""
