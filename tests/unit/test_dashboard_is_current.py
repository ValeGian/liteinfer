"""The committed dashboard must be what the committed results render to.

`docs/index.html` is generated from `benchmarks/results/` by `bench report
--out`, and it is committed so GitHub Pages can serve it. That makes it the one
file in the repo that silently goes stale: a run lands, the JSON is committed,
and the published page keeps showing the previous engine. It went 16 PRs out of
date that way. This test is the guard — it needs no GPU, because rendering the
report is a pure transform of data already in the tree.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks import report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD = _REPO_ROOT / "docs" / "index.html"
_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"

_REGENERATE = "bench report --out docs/index.html"


def test_the_published_dashboard_matches_the_stored_results() -> None:
    expected = report.as_html(report.load_results(_RESULTS_DIR))

    assert _DASHBOARD.read_text(encoding="utf-8") == expected, (
        f"docs/index.html is stale — regenerate it with `{_REGENERATE}`"
    )
