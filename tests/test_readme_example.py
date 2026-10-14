"""The README's minimal agent example runs as written (offline, light embedder)."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _block_after(marker: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(marker)
    match = re.search(r"```python\n(.*?)```", text[start:], re.S)
    assert match, f"no python block after {marker!r}"
    return match.group(1)


def test_minimal_agent_example_runs():
    code = _block_after("A minimal runnable example")
    namespace: dict = {}
    exec(compile(code, str(README), "exec"), namespace)  # noqa: S102
    assert namespace["result"].results[0].name == "fetch_url"
    assert namespace["nothing"].abstained
