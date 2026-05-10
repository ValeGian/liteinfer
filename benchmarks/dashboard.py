"""Generate a self-contained interactive HTML dashboard from a benchmark history JSONL file.

Layout: one section per workload (most recent run). Each section is a table with
engines as rows and metrics as columns.
- Best cell per column: green background.
- Worst cell per column: red background.
- Every non-baseline cell shows an inline x ratio vs the first (baseline) engine.
- Metric column headers and engine name cells carry CSS tooltips with context.

Default output is docs/dashboard.html so the file is tracked by git and
accessible via htmlpreview.github.io.

Usage:
    python -m benchmarks.dashboard \\
        --history benchmarks/results/history.jsonl \\
        --output docs/dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Metrics shown per workload. (key, display_label, higher_is_better, tooltip)
_THROUGHPUT_METRICS: list[tuple[str, str, bool, str]] = [
    ("requests_per_second", "req/s", True,
     "Requests completed per wall-second. Measured over the full batch from first submission to last completion. Higher is better."),
    ("output_tokens_per_second", "tok/s", True,
     "Output tokens emitted per wall-second across all requests. Higher is better."),
    ("e2e_latency_p50_s", "E2E p50", False,
     "Median end-to-end latency per request, from batch submission (t₀) to last token. Includes queue wait time — later requests in a B=1 queue have higher E2E. Lower is better."),
    ("e2e_latency_p99_s", "E2E p99", False,
     "99th-percentile end-to-end latency. With 32 requests and B=1, p99 ≈ the last request's E2E. Lower is better."),
]

_LATENCY_METRICS: list[tuple[str, str, bool, str]] = [
    ("ttft_p50_s", "TTFT p50", False,
     "Median time-to-first-token: wall clock from request submission to when the first output token is ready. No queue contamination in this workload. Lower is better."),
    ("ttft_p99_s", "TTFT p99", False,
     "99th-percentile TTFT. Variance here reflects scheduling jitter and GPU state. Lower is better."),
    ("e2e_latency_p50_s", "E2E p50", False,
     "Median end-to-end latency per request: submission to last token. Sequential workload — no queue wait. Lower is better."),
    ("output_tokens_per_second", "tok/s", True,
     "Output tokens per wall-second (total tokens / total wall time across all sequential requests). Reflects steady-state decode throughput. Higher is better."),
]

_WORKLOAD_METRICS: dict[str, list[tuple[str, str, bool, str]]] = {
    "throughput": _THROUGHPUT_METRICS,
    "latency": _LATENCY_METRICS,
    "prefix_share": _THROUGHPUT_METRICS,
}

_LATENCY_KEYS = {"ttft_p50_s", "ttft_p99_s", "e2e_latency_p50_s", "e2e_latency_p99_s"}

_WORKLOAD_SUBTITLES: dict[str, str] = {
    "throughput": "All requests submitted at once — engine queues them, processes B=1 at a time. E2E includes queue wait.",
    "latency": "Sequential, no queue — each request sent only after previous finishes. Pure per-request engine latency.",
    "prefix_share": "Prompts share a long common prefix. Designed to stress prefix caching.",
}

_ENGINE_TIPS: dict[str, str] = {
    "liteinfer": (
        "liteinfer · cache_mode=none (RECOMPUTE), max_num_seqs=1\n"
        "Every step re-feeds the full and growing sequence. "
        "No KV cache — decode cost grows linearly with sequence length."
    ),
    "liteinfer-kvcache": (
        "liteinfer · cache_mode=eager (DynamicCache), max_num_seqs=1\n"
        "Prefill runs once to populate the KV cache. "
        "Each decode step passes only the new token — O(1) input, O(n) attention lookup."
    ),
    "liteinfer-native-kvcache": (
        "liteinfer · cache_mode=native_eager (plain-tensor KV cache), max_num_seqs=1\n"
        "Prefill runs once; each decode step passes only the new token. "
        "KV cache stored as plain tensors — no DynamicCache wrapper overhead."
    ),
    "liteinfer-paged-kvcache": (
        "liteinfer · cache_mode=paged (block-allocated KV cache), max_num_seqs=1\n"
        "Tokens stored in fixed-size blocks drawn from a pre-allocated pool. "
        "Eliminates memory fragmentation; foundation for prefix sharing (§2.2)."
    ),
    "liteinfer-b4": (
        "liteinfer · cache_mode=eager, max_num_seqs=4\n"
        "Static batching with B=4: up to four prompts share one PREFILL "
        "and decode together until every member finishes. Eager KV cache; "
        "left-padded prefill with pad-aware additive attention mask."
    ),
    "vllm": (
        "vLLM 0.20.0 · max_num_seqs=1 (B=1)\n"
        "FlashAttention 2, torch.compile, CUDA graphs for decode, "
        "paged KV cache, prefix caching enabled."
    ),
    "vllm-b4": (
        "vLLM 0.20.0 · max_num_seqs=4 (B=4)\n"
        "Continuous batching of up to 4 sequences. Same kernels as vllm B=1: "
        "FlashAttention 2, torch.compile, CUDA graphs, paged KV, prefix caching."
    ),
}


_DELTA_THRESHOLD = 0.05


def _delta_class(current: float, previous: float | None, higher_is_better: bool) -> str:
    """Return CSS class reflecting change from *previous* to *current*.

    Returns "ok" for improvements > threshold, "bad" for regressions > threshold,
    and "" when the delta is within threshold or there is no valid baseline.
    """
    if not previous:
        return ""
    ratio = current / previous if higher_is_better else previous / current
    if ratio > 1 + _DELTA_THRESHOLD:
        return "ok"
    if ratio < 1 - _DELTA_THRESHOLD:
        return "bad"
    return ""


def load_history(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_html(runs: list[dict]) -> str:
    latest: dict[str, dict] = {}
    previous: dict[str, dict] = {}
    for run in runs:
        workload = run["workload"]
        if workload in latest:
            previous[workload] = latest[workload]
        latest[workload] = run

    sections = "\n".join(
        _workload_section(run, previous.get(run["workload"])) for run in latest.values()
    )
    n_engines = max((len(r["results"]) for r in latest.values()), default=0)
    meta = (
        f"{len(runs)} run(s) in history &nbsp;·&nbsp; "
        f"{n_engines} engine(s) compared &nbsp;·&nbsp; "
        f"<span class='legend-pill best-pill'>green = best per column</span> "
        f"<span class='legend-pill worst-pill'>red = worst per column</span> "
        f"&nbsp;·&nbsp; x ratio is vs first (baseline) engine"
    )
    return _page(meta, sections)


def _workload_section(run: dict, prev_run: dict | None = None) -> str:
    workload = run["workload"]
    metrics = _WORKLOAD_METRICS.get(workload, _THROUGHPUT_METRICS)
    subtitle = _WORKLOAD_SUBTITLES.get(workload, "")
    num_prompts = run.get("num_prompts", "?")
    tag = run.get("tag", "")
    model = run.get("model", "")
    model_short = model.split("/")[-1]
    ts = run.get("timestamp", "")
    sequential = run.get("sequential", False)
    submission = "sequential, no queue" if sequential else "all submitted at once"

    meta_line = (
        f"<span class='meta-item'>📅 {ts}</span>"
        + (f"<span class='meta-item tag-pill'>{tag}</span>" if tag else "")
        + f"<span class='meta-item' title='{model}'>🤖 {model_short}</span>"
        + f"<span class='meta-item'>{num_prompts} prompts · {submission}</span>"
    )

    table = _comparison_table(run, metrics, prev_run)
    return (
        f"<section>"
        f"<h2>{workload}</h2>"
        f"<p class='subtitle'>{subtitle}</p>"
        f"<div class='run-meta'>{meta_line}</div>"
        f"{table}"
        f"</section>"
    )


def _comparison_table(
    run: dict, metrics: list[tuple[str, str, bool, str]], prev_run: dict | None = None
) -> str:
    engines = [e["engine"] for e in run["results"]]
    by_engine = {e["engine"]: e for e in run["results"]}
    prev_by_engine = {e["engine"]: e for e in prev_run["results"]} if prev_run else {}

    # Baseline = first engine.
    baseline_engine = engines[0] if engines else None
    baseline_data = by_engine.get(baseline_engine, {}) if baseline_engine else {}

    # Per-column best/worst values.
    col_extremes: dict[str, tuple[float, float]] = {}
    for key, _, higher_is_better, _ in metrics:
        vals = [by_engine[e][key] for e in engines if by_engine[e].get(key) is not None]
        if len(vals) < 2:
            continue
        col_extremes[key] = (
            (max(vals), min(vals)) if higher_is_better else (min(vals), max(vals))
        )

    # Header row with tooltips on metric labels.
    metric_headers = "".join(
        f'<th><span class="tip-host" data-tip="{_escape(tip)}">{label}</span></th>'
        for _, label, _, tip in metrics
    )
    batch_tip = (
        "max_num_seqs the engine was configured with. B=1 disables batching; "
        "B=N lets the scheduler run up to N sequences in one forward pass."
    )
    thead = (
        f"<thead><tr><th class='l'>Engine</th>"
        f"<th><span class='tip-host' data-tip='{_escape(batch_tip)}'>B</span></th>"
        f"{metric_headers}</tr></thead>"
    )

    # Data rows.
    rows: list[str] = []
    for engine in engines:
        data = by_engine[engine]
        is_baseline = engine == baseline_engine
        prev_data = prev_by_engine.get(engine, {})

        engine_tip = _ENGINE_TIPS.get(engine, "")
        engine_cell = (
            f'<td class="engine"><span class="tip-host" data-tip="{_escape(engine_tip)}">'
            f'{engine}</span></td>'
        )
        batch_size = data.get("batch_size", 1)
        batch_cell = f'<td class="batch">{batch_size}</td>'

        cells: list[str] = [engine_cell, batch_cell]
        for key, _, higher_is_better, _ in metrics:
            val = data.get(key)
            if val is None:
                cells.append("<td>&mdash;</td>")
                continue

            text = _fmt(key, val)

            # Inline ratio vs baseline.
            baseline_val = baseline_data.get(key)
            ratio_html = ""
            if is_baseline:
                ratio_html = "<small class='ratio-base'>base</small>"
            elif baseline_val and baseline_val != 0:
                ratio = val / baseline_val if higher_is_better else baseline_val / val
                ratio_cls = "ratio-better" if ratio >= 1.0 else "ratio-worse"
                ratio_html = f"<small class='{ratio_cls}'>{ratio:.2f}x</small>"

            # Best/worst cell coloring + tooltip; fall back to cross-run delta class.
            extremes = col_extremes.get(key)
            css = ""
            cell_tip = ""
            if extremes:
                best, worst = extremes
                if val == best:
                    css = "best"
                    if worst != 0:
                        gap = (best / worst) if higher_is_better else (worst / best)
                        cell_tip = f"Best in column — {gap:.2f}x better than worst"
                elif val == worst:
                    css = "worst"
                    if best != 0:
                        gap = (best / worst) if higher_is_better else (worst / best)
                        cell_tip = f"Worst in column — {gap:.2f}x slower than best"
            if not css:
                css = _delta_class(val, prev_data.get(key), higher_is_better)

            inner = f'{text}{ratio_html}'
            if cell_tip:
                inner = f'<span class="tip-host" data-tip="{_escape(cell_tip)}">{inner}</span>'
            cells.append(f'<td class="{css}">{inner}</td>' if css else f"<td>{inner}</td>")

        rows.append(f"<tr>{''.join(cells)}</tr>")

    tbody = "\n".join(rows)
    return f"<div class='wrap'><table>{thead}<tbody>{tbody}</tbody></table></div>"


def _fmt(key: str, value: float) -> str:
    if key in _LATENCY_KEYS:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f}"


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace("\n", "&#10;")


def _page(meta: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiteInfer Benchmark Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: #f4f4f6;
    color: #1a1a1a;
    padding: 2rem 2.5rem;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: .3rem }}
  .page-meta {{
    font-size: .8rem; color: #555; margin-bottom: 2rem;
    display: flex; flex-wrap: wrap; align-items: center; gap: .5rem;
  }}
  section {{ margin-bottom: 2.8rem }}
  h2 {{
    font-size: 1.05rem; font-weight: 700;
    text-transform: capitalize; letter-spacing: .01em;
    margin-bottom: .2rem;
  }}
  .subtitle {{ font-size: .82rem; color: #555; margin-bottom: .4rem }}
  .run-meta {{
    display: flex; flex-wrap: wrap; gap: .5rem;
    margin-bottom: .75rem;
  }}
  .meta-item {{
    font-size: .76rem; color: #666;
    background: #ebebeb; border-radius: 4px;
    padding: 2px 8px;
  }}
  .wrap {{ overflow-x: auto }}
  table {{
    border-collapse: collapse;
    font-size: .83rem;
    white-space: nowrap;
    min-width: 360px;
  }}
  th {{
    background: #e4e4e7;
    font-weight: 600;
    padding: 7px 14px;
    border: 1px solid #ccc;
    text-align: right;
  }}
  th.l {{ text-align: left }}
  td {{
    padding: 6px 14px;
    border: 1px solid #ddd;
    text-align: right;
    background: #fff;
    vertical-align: middle;
  }}
  td.engine {{
    text-align: left;
    font-weight: 600;
    background: #fafafa;
    font-family: ui-monospace, monospace;
    font-size: .8rem;
  }}
  .best {{ background: #d1fae5; color: #065f46; font-weight: 700 }}
  .worst {{ background: #fee2e2; color: #991b1b; font-weight: 700 }}
  small.ratio-base  {{ display:block; font-size:.68rem; color:#999; font-weight:400 }}
  small.ratio-better {{ display:block; font-size:.68rem; color:#059669; font-weight:600 }}
  small.ratio-worse  {{ display:block; font-size:.68rem; color:#dc2626; font-weight:600 }}

  /* ── Tooltip ── */
  .tip-host {{
    position: relative;
    cursor: help;
    text-decoration: underline dotted #aaa;
    text-underline-offset: 2px;
  }}
  .tip-host::after {{
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #18181b;
    color: #f4f4f5;
    font-size: .74rem;
    font-weight: 400;
    padding: 6px 10px;
    border-radius: 6px;
    white-space: pre-wrap;
    max-width: 280px;
    width: max-content;
    pointer-events: none;
    opacity: 0;
    transition: opacity .15s ease;
    z-index: 200;
    text-align: left;
    box-shadow: 0 4px 12px rgba(0,0,0,.25);
    line-height: 1.45;
  }}
  .tip-host:hover::after {{ opacity: 1 }}
  td.engine .tip-host::after {{ left: 0; transform: none }}
  tbody tr:first-child td.engine .tip-host::after {{ bottom: auto; top: calc(100% + 8px) }}
  thead th .tip-host::after {{ bottom: auto; top: calc(100% + 8px); left: 50%; transform: translateX(-50%) }}

  /* ── Legend pills ── */
  .legend-pill {{
    display: inline-block;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: .76rem;
    font-weight: 600;
  }}
  .best-pill {{ background: #d1fae5; color: #065f46 }}
  .worst-pill {{ background: #fee2e2; color: #991b1b }}
  .tag-pill {{
    background: #dbeafe;
    color: #1d4ed8;
    font-weight: 600;
  }}
</style>
</head>
<body>
<h1>LiteInfer Benchmark Dashboard</h1>
<div class="page-meta">{meta}</div>
{body}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML dashboard from a benchmark history JSONL file."
    )
    parser.add_argument("--history", type=Path, required=True, help="Path to history.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/dashboard.html"),
        help="Path to write dashboard HTML (default: docs/dashboard.html).",
    )
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
