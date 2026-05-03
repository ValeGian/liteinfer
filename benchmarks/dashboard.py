"""Generate a self-contained HTML dashboard from a benchmark history JSONL file.

Each line of the history file is a JSON object produced by `compare.py --append-history`.
Rows are ordered chronologically; cells are color-coded against the previous run for the
same engine so regressions and improvements stand out immediately.

Usage:
    python -m benchmarks.dashboard \\
        --history benchmarks/results/history.jsonl \\
        --output benchmarks/results/dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (key, display label, higher_is_better)
_METRIC_DEFS: list[tuple[str, str, bool]] = [
    ("requests_per_second", "req/s", True),
    ("output_tokens_per_second", "tok/s", True),
    ("ttft_p50_s", "TTFT p50", False),
    ("ttft_p99_s", "TTFT p99", False),
    ("e2e_latency_p50_s", "E2E p50", False),
    ("e2e_latency_p99_s", "E2E p99", False),
]

_LATENCY_KEYS = {"ttft_p50_s", "ttft_p99_s", "e2e_latency_p50_s", "e2e_latency_p99_s"}

_CHANGE_THRESHOLD = 0.05  # 5 % swing required to color a cell


def load_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_html(runs: list[dict]) -> str:
    workloads = sorted({r["workload"] for r in runs})
    all_engines = sorted({e["engine"] for r in runs for e in r["results"]})

    ts_range = f"{runs[0]['timestamp']} &rarr; {runs[-1]['timestamp']}" if runs else ""
    sections = "\n".join(
        _workload_section(workload, [r for r in runs if r["workload"] == workload], all_engines)
        for workload in workloads
    )
    return _page(f"{len(runs)} runs &nbsp;&middot;&nbsp; {ts_range}", sections)


def _workload_section(workload: str, runs: list[dict], all_engines: list[str]) -> str:
    thead = _thead(all_engines)
    rows: list[str] = []
    prev_by_engine: dict[str, dict] = {}
    for run in runs:
        by_engine = {e["engine"]: e for e in run["results"]}
        rows.append(_trow(run, by_engine, prev_by_engine, all_engines))
        prev_by_engine = by_engine
    tbody = "\n".join(rows)
    return (
        f"<h2>Workload: {workload}</h2>"
        f"<div class='wrap'><table>{thead}<tbody>{tbody}</tbody></table></div>"
    )


def _thead(all_engines: list[str]) -> str:
    n = len(_METRIC_DEFS)
    engine_cols = "".join(f'<th colspan="{n}">{e}</th>' for e in all_engines)
    metric_cols = "".join(
        f"<th>{label}</th>"
        for _ in all_engines
        for _, label, _ in _METRIC_DEFS
    )
    return (
        "<thead>"
        f"<tr>"
        f"<th rowspan='2' class='l'>Tag</th>"
        f"<th rowspan='2' class='l'>Timestamp</th>"
        f"<th rowspan='2' class='l'>Model</th>"
        f"{engine_cols}"
        f"</tr>"
        f"<tr>{metric_cols}</tr>"
        "</thead>"
    )


def _trow(
    run: dict,
    by_engine: dict[str, dict],
    prev_by_engine: dict[str, dict],
    all_engines: list[str],
) -> str:
    tag = run.get("tag") or ""
    tag_html = f'<span class="tag">{tag}</span>' if tag else "&mdash;"
    model = run.get("model", "").split("/")[-1]

    cells: list[str] = []
    for engine in all_engines:
        cur = by_engine.get(engine)
        prev = prev_by_engine.get(engine)
        for key, _, higher_is_better in _METRIC_DEFS:
            if cur is None or cur.get(key) is None:
                cells.append("<td>&mdash;</td>")
                continue
            val: float = cur[key]
            text = _fmt(key, val)
            css = _delta_class(val, prev.get(key) if prev else None, higher_is_better)
            cells.append(f'<td class="{css}">{text}</td>' if css else f"<td>{text}</td>")

    cells_html = "".join(cells)
    return (
        f"<tr>"
        f"<td class='l'>{tag_html}</td>"
        f"<td class='l'>{run.get('timestamp', '')}</td>"
        f"<td class='l'>{model}</td>"
        f"{cells_html}"
        f"</tr>"
    )


def _fmt(key: str, value: float) -> str:
    if key in _LATENCY_KEYS:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f}"


def _delta_class(current: float, previous: float | None, higher_is_better: bool) -> str:
    if previous is None or previous == 0:
        return ""
    ratio = current / previous
    improved = ratio >= 1 + _CHANGE_THRESHOLD if higher_is_better else ratio <= 1 - _CHANGE_THRESHOLD
    worsened = ratio <= 1 - _CHANGE_THRESHOLD if higher_is_better else ratio >= 1 + _CHANGE_THRESHOLD
    if improved:
        return "ok"
    if worsened:
        return "bad"
    return ""


def _page(meta: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LiteInfer Benchmark Dashboard</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:2rem;background:#f5f5f5;color:#1a1a1a}}
  h1{{font-size:1.4rem;font-weight:700;margin:0 0 .2rem}}
  h2{{font-size:1rem;font-weight:600;margin:2rem 0 .6rem;padding-bottom:.3rem;border-bottom:2px solid #ddd}}
  .meta{{color:#666;font-size:.82rem;margin-bottom:1.8rem}}
  .wrap{{overflow-x:auto}}
  table{{border-collapse:collapse;font-size:.8rem;white-space:nowrap}}
  th{{background:#e8e8e8;font-weight:600;padding:5px 11px;border:1px solid #ccc;text-align:right}}
  th.l{{text-align:left}}
  td{{padding:4px 11px;border:1px solid #e4e4e4;text-align:right;background:#fff}}
  td.l{{text-align:left}}
  tr:hover td{{background:#eef2ff}}
  .ok{{color:#1a7f37;font-weight:600}}
  .bad{{color:#cf222e;font-weight:600}}
  .tag{{background:#dbeafe;color:#1d4ed8;border-radius:3px;padding:1px 6px;font-size:.74rem;font-weight:500}}
</style>
</head>
<body>
<h1>LiteInfer Benchmark Dashboard</h1>
<p class="meta">{meta}</p>
{body}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML dashboard from a benchmark history JSONL file."
    )
    parser.add_argument("--history", type=Path, required=True, help="Path to history.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to write dashboard.html")
    args = parser.parse_args()

    runs = load_history(args.history)
    if not runs:
        print(f"No runs found in {args.history}")
        return

    html = build_html(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Dashboard written to {args.output}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
