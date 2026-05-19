"""Dashboard builder: reads result JSON files and emits self-contained HTML.

All output HTML is fully self-contained:
- Result data embedded as window.__BENCH_DATA__ in a <script> tag
- All CSS in a <style> tag
- All JS inlined in <script> tags
- No CDN links or external fetches

This ensures the dashboard works as file:///path/to/docs/index.html and on
GitHub Pages with no additional infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_result(run_id: str, results_dir: Path) -> dict | None:
    result_file = results_dir / f"{run_id}.json"
    if not result_file.exists():
        return None
    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


def _group_results(results: list[dict]) -> dict[tuple, list[dict]]:
    """Group results into tabs by (benchmark_type, model, target_isl, target_osl)."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (
            r.get("benchmark_type", ""),
            r.get("model", ""),
            r["dataset"].get("target_isl", 0),
            r["dataset"].get("target_osl", 0),
        )
        groups.setdefault(key, []).append(r)
    return groups


def _model_short_name(model: str) -> str:
    return model.split("/")[-1]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f8f9fa;
    color: #212529;
    padding: 24px;
}
h1 { font-size: 1.6rem; margin-bottom: 20px; color: #1a1a2e; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.tab-btn {
    padding: 8px 16px;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
    font-size: 0.85rem;
    color: #495057;
    transition: background 0.15s;
}
.tab-btn.active { background: #1a1a2e; color: #fff; border-color: #1a1a2e; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.note {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 0.83rem;
    color: #664d03;
}
table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
th {
    background: #1a1a2e;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
}
td {
    padding: 10px 14px;
    font-size: 0.85rem;
    border-bottom: 1px solid #f0f0f0;
    white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
.best { background: #d4edda; color: #155724; font-weight: 600; }
.worst { background: #f8d7da; color: #721c24; }
.tag { font-size: 0.75rem; color: #6c757d; margin-left: 6px; }
"""

_JS = """
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('btn-' + tabId).classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
}
window.addEventListener('DOMContentLoaded', function() {
    var first = document.querySelector('.tab-btn');
    if (first) first.click();
});
"""


def _throughput_table(results: list[dict]) -> str:
    # Columns: Engine, tok/s, req/s, TTFT p50, TTFT p99, ITL p50, ITL p99
    columns = [
        ("Engine", None, True),
        ("tok/s", "tokens_per_second", False),
        ("req/s", "requests_per_second", False),
        ("TTFT p50 (ms)", "ttft_p50_ms", False),
        ("TTFT p99 (ms)", "ttft_p99_ms", False),
        ("ITL p50 (ms)", "itl_p50_ms", False),
        ("ITL p99 (ms)", "itl_p99_ms", False),
    ]
    return _render_table(results, columns, higher_is_better={"tokens_per_second", "requests_per_second"})


def _latency_table(results: list[dict]) -> str:
    # Columns: Engine, TTFT p50, TTFT p95, TTFT p99, ITL p50, ITL p95, E2E p50
    columns = [
        ("Engine", None, True),
        ("TTFT p50 (ms)", "ttft_p50_ms", False),
        ("TTFT p95 (ms)", "ttft_p95_ms", False),
        ("TTFT p99 (ms)", "ttft_p99_ms", False),
        ("ITL p50 (ms)", "itl_p50_ms", False),
        ("ITL p95 (ms)", "itl_p95_ms", False),
        ("E2E p50 (ms)", "e2e_p50_ms", False),
    ]
    return _render_table(results, columns, higher_is_better=set())


def _render_table(
    results: list[dict],
    columns: list[tuple],
    higher_is_better: set[str],
) -> str:
    # Collect metric values to find best/worst per column
    metric_values: dict[str, list[float]] = {}
    for _, key, is_label in columns:
        if key is None or is_label:
            continue
        vals = []
        for r in results:
            v = r.get("summary", {}).get(key)
            if v is not None:
                vals.append(float(v))
        metric_values[key] = vals

    def _cell_class(key: str, value: float | None) -> str:
        if value is None or key is None:
            return ""
        vals = metric_values.get(key, [])
        if len(vals) < 2:
            return ""
        if key in higher_is_better:
            if value == max(vals):
                return "best"
            if value == min(vals):
                return "worst"
        else:
            if value == min(vals):
                return "best"
            if value == max(vals):
                return "worst"
        return ""

    header = "".join(f"<th>{col[0]}</th>" for col in columns)
    rows = []
    for r in results:
        engine = r.get("engine", "?")
        tag = r.get("tag", "")
        tag_html = f'<span class="tag">[{tag}]</span>' if tag else ""
        cells = [f"<td>{engine}{tag_html}</td>"]
        for _, key, is_label in columns:
            if is_label:
                continue
            v = r.get("summary", {}).get(key) if key else None
            if v is not None:
                css_class = _cell_class(key, float(v))
                formatted = f"{float(v):.1f}"
                class_attr = f' class="{css_class}"' if css_class else ""
                cells.append(f"<td{class_attr}>{formatted}</td>")
            else:
                cells.append("<td>—</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table>"
        "<thead><tr>" + header + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


_TTFT_THROUGHPUT_NOTE = (
    "TTFT in throughput mode reflects scheduling contention and is not comparable to "
    "TTFT from a latency-mode run. Compare TTFT only within runs of the same benchmark type."
)


def _build_html(groups: dict[tuple, list[dict]], title: str = "Benchmark Dashboard") -> str:
    tab_buttons = []
    tab_panels = []

    for tab_idx, (key, results) in enumerate(groups.items()):
        benchmark_type, model, isl, osl = key
        model_short = _model_short_name(model)
        tab_label = f"{benchmark_type} · {model_short} · ISL={isl} OSL={osl}"
        tab_id = f"tab{tab_idx}"

        tab_buttons.append(
            f'<button class="tab-btn" id="btn-{tab_id}" '
            f'onclick="switchTab(\'{tab_id}\')">{tab_label}</button>'
        )

        if benchmark_type == "throughput":
            note_html = f'<div class="note">{_TTFT_THROUGHPUT_NOTE}</div>'
            table_html = _throughput_table(results)
        else:
            note_html = ""
            table_html = _latency_table(results)

        tab_panels.append(
            f'<div class="tab-panel" id="{tab_id}">'
            f"{note_html}{table_html}"
            f"</div>"
        )

    tabs_html = '<div class="tabs">' + "".join(tab_buttons) + "</div>"
    panels_html = "".join(tab_panels)

    no_data_html = ""
    if not groups:
        no_data_html = "<p>No pinned results. Use <code>bench dashboard promote &lt;run_id&gt;</code> to add results.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>{title}</h1>
{tabs_html}
{panels_html}
{no_data_html}
<script>
{_JS}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_comparison_dashboard(
    result_ids: list[str],
    results_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Build a comparison dashboard from the given result IDs."""
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    results = []
    for run_id in result_ids:
        r = _load_result(run_id, results_dir)
        if r is not None:
            results.append(r)

    groups = _group_results(results)
    html = _build_html(groups, title="Benchmark Comparison")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_main_dashboard(
    results_dir: str | Path = "benchmarks/results",
    output_path: str | Path = "docs/index.html",
) -> None:
    """Build the main dashboard from pinned results in main.json."""
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    main_json = results_dir / "main.json"
    if main_json.exists():
        with open(main_json, encoding="utf-8") as f:
            index = json.load(f)
        pinned_ids = index.get("pinned", [])
    else:
        pinned_ids = []

    build_comparison_dashboard(pinned_ids, results_dir, output_path)
