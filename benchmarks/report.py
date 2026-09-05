"""Turns result files into a comparison table, as text or as a standalone page.

Results are grouped by (mode, model, ISL, OSL) — only runs inside one group are
comparable. Within a group each config is scored three ways: against the config
it improves on, against the root of its lineage, and against vLLM at the same
batch width.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks.configs import CONFIGS

# (key, header, lower_is_better); the first entry of each mode is its headline.
COLUMNS: dict[str, list[tuple[str, str, bool]]] = {
    "throughput": [
        ("output_tokens_per_s", "tok/s", False),
        ("requests_per_s", "req/s", False),
        ("wall_time_s", "wall (s)", True),
    ],
    "latency": [
        ("itl_p50_ms", "ITL p50", True),
        ("itl_p95_ms", "ITL p95", True),
        ("ttft_p50_ms", "TTFT p50", True),
        ("ttft_p95_ms", "TTFT p95", True),
        ("e2e_p50_ms", "E2E p50", True),
    ],
}

MODE_NOTE = {
    "throughput": "Every request offered at once. Headline metric: output tok/s.",
    "latency": "One request in flight at a time. Headline metric: ITL p50.",
}

_ORDER = {name: index for index, name in enumerate(CONFIGS)}


def load_results(results_dir: str | Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(results_dir).glob("*.json"))
    ]


def group(results: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for result in results:
        key = (
            result["mode"],
            result["model"],
            result["dataset"]["target_isl"],
            result["dataset"]["target_osl"],
        )
        grouped.setdefault(key, []).append(result)
    return grouped


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(result: dict, reference: dict | None, mode: str) -> float | None:
    """Headline-metric ratio against ``reference``. Above 1 means better."""
    if reference is None or reference is result:
        return None
    key, _, lower_is_better = COLUMNS[mode][0]
    mine, theirs = result["summary"][key], reference["summary"][key]
    if not mine or not theirs:
        return None
    return theirs / mine if lower_is_better else mine / theirs


def _baseline(result: dict, peers: dict[str, dict]) -> dict | None:
    return peers.get(result.get("baseline") or "")


def _lineage_root(result: dict, peers: dict[str, dict]) -> dict | None:
    config = CONFIGS.get(result["config"])
    while config is not None and config.baseline is not None:
        config = CONFIGS[config.baseline]
    return peers.get(config.name) if config else None


def _vllm_at_same_width(result: dict, members: list[dict]) -> dict | None:
    """Matching on batch width is the point: comparing a B=1 engine to a B=32 one
    measures the batch width, not the engine."""
    if result["engine"] == "vllm":
        return None
    return next(
        (
            other
            for other in members
            if other["engine"] == "vllm" and other["max_num_seqs"] == result["max_num_seqs"]
        ),
        None,
    )


@dataclass(frozen=True)
class Row:
    result: dict
    base: str | None
    vs_base: float | None
    vs_root: float | None
    vs_vllm: float | None

    def value(self, key: str) -> float:
        return self.result["summary"][key]


def rows(members: list[dict], mode: str) -> list[Row]:
    """One scored row per result, ordered so each config follows its baseline."""
    peers = {r["config"]: r for r in members}
    ordered = sorted(members, key=lambda r: _ORDER.get(r["config"], len(_ORDER)))
    return [
        Row(
            result=result,
            base=result.get("baseline"),
            vs_base=_score(result, _baseline(result, peers), mode),
            vs_root=_score(result, _lineage_root(result, peers), mode),
            vs_vllm=_score(result, _vllm_at_same_width(result, members), mode),
        )
        for result in ordered
    ]


def _mismatched(members: list[dict]) -> bool:
    return len({r["dataset"]["sha256"] for r in members}) > 1


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return f"{value:.2f}x" if value else "-"


def as_text(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines: list[str] = []
    for (mode, model, isl, osl), members in sorted(group(results).items()):
        columns = COLUMNS[mode]
        lines.append(
            f"\n{mode}  |  {model.split('/')[-1]}  |  ISL={isl} OSL={osl}"
            f"  |  {members[0]['dataset']['num_samples']} prompts"
        )
        if _mismatched(members):
            lines.append("  WARNING: these runs did not all use the same prompts")
        header = (
            f"{'config':<27}{'improves on':<25}"
            + "".join(f"{head:>11}" for _, head, _ in columns)
            + f"{'vs base':>9}{'vs first':>9}{'vs vLLM':>9}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows(members, mode):
            values = "".join(f"{row.value(k):>11,.1f}" for k, _, _ in columns)
            name = row.result["config"] + ("*" if _is_historical(row) else "")
            lines.append(
                f"{name:<27}{row.base or '-':<25}{values}"
                f"{_fmt(row.vs_base):>9}{_fmt(row.vs_root):>9}{_fmt(row.vs_vllm):>9}"
            )
    if any(_is_historical(row) for members, mode in _groups(results) for row in rows(members, mode)):
        lines.append("\n* measured before the code was removed; no longer runnable")
    return "\n".join(lines)


def _groups(results: list[dict]):
    return [(members, key[0]) for key, members in sorted(group(results).items())]


def _is_historical(row: Row) -> bool:
    config = CONFIGS.get(row.result["config"])
    return bool(config and config.historical)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_STYLE = """
:root { --bg:#fbfbfd; --fg:#17171a; --muted:#6b7280; --line:#e7e7ec; --card:#fff;
        --good:#0a7f5c; --bad:#b4342a; --bar:#c9d3e8; --bar-vllm:#e2d6c4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111114; --fg:#eaeaef; --muted:#9aa0aa; --line:#2b2b32; --card:#191920;
          --good:#4ade80; --bad:#f87171; --bar:#33405e; --bar-vllm:#4a4032; }
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--fg); padding:44px 22px 72px;
       font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }
main { max-width:1120px; margin:0 auto; }
h1 { font-size:1.55rem; letter-spacing:-.021em; }
.lede { color:var(--muted); font-size:.92rem; margin:8px 0 26px; max-width:70ch; }
.facts { display:flex; flex-wrap:wrap; gap:8px 26px; padding:14px 18px; margin-bottom:34px;
         background:var(--card); border:1px solid var(--line); border-radius:10px;
         font-size:.8rem; color:var(--muted); }
.facts b { color:var(--fg); font-weight:600; }
h2 { font-size:1.02rem; margin:34px 0 3px; letter-spacing:-.01em; }
.meta { color:var(--muted); font-size:.81rem; margin-bottom:12px; }
.warn { color:var(--bad); font-size:.81rem; margin-bottom:12px; font-weight:600; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--card);
        border:1px solid var(--line); border-radius:10px; }
th,td { padding:8px 13px; text-align:right; white-space:nowrap;
        font-variant-numeric:tabular-nums; }
th:first-child,td:first-child { text-align:left; }
th { font-size:.71rem; font-weight:600; color:var(--muted); text-transform:uppercase;
     letter-spacing:.05em; border-bottom:1px solid var(--line); }
td { border-bottom:1px solid var(--line); font-size:.86rem; }
tr:last-child td { border-bottom:none; }
tr.ref td:first-child { color:var(--muted); }
.name { position:relative; }
.name .label { font-weight:500; }
.name .desc { display:block; color:var(--muted); font-size:.74rem; font-weight:400; }
.bar { display:block; height:3px; border-radius:2px; background:var(--bar); margin-top:5px; }
tr.ref .bar { background:var(--bar-vllm); }
td.best { color:var(--good); font-weight:600; }
.up { color:var(--good); font-weight:600; }
.down { color:var(--bad); font-weight:600; }
.nil { color:var(--muted); }
footer { color:var(--muted); font-size:.78rem; margin-top:38px; }
"""


def _delta(value: float | None) -> str:
    if value is None:
        return '<td class="nil">—</td>'
    return f'<td class="{"up" if value >= 1 else "down"}">{value:.2f}x</td>'


def _table(members: list[dict], mode: str) -> str:
    columns = COLUMNS[mode]
    scored = rows(members, mode)
    best = {
        key: (min if lower else max)(row.value(key) for row in scored) for key, _, lower in columns
    }
    head_key, _, head_lower = COLUMNS[mode][0]
    values = [row.value(head_key) for row in scored]
    # Bar length shows standing on the headline metric; invert when lower is better.
    scale = [(min(values) / v if head_lower else v / max(values)) for v in values]

    body = []
    for row, fraction in zip(scored, scale, strict=True):
        result = row.result
        cells = []
        for key, _, _ in columns:
            value = result["summary"][key]
            css = ' class="best"' if value == best[key] else ""
            cells.append(f"<td{css}>{value:,.1f}</td>")
        lineage = f"improves on {row.base}" if row.base else "reference point"
        if _is_historical(row):
            lineage += " &middot; removed from the codebase"
        body.append(
            f'<tr class="{"ref" if result["engine"] == "vllm" else ""}">'
            f'<td class="name"><span class="label">{html.escape(result["config"])}</span>'
            f'<span class="desc">{html.escape(result["description"])} '
            f"&middot; {html.escape(lineage)}</span>"
            f'<span class="bar" style="width:{max(fraction, 0.012) * 100:.1f}%"></span></td>'
            + "".join(cells)
            + _delta(row.vs_base)
            + _delta(row.vs_root)
            + _delta(row.vs_vllm)
            + "</tr>"
        )

    heads = "".join(f"<th>{head}</th>" for _, head, _ in columns)
    return (
        '<div class="scroll"><table><thead><tr><th>config</th>'
        f"{heads}<th>vs base</th><th>vs first</th><th>vs vLLM</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _facts(results: list[dict]) -> str:
    first = results[0]
    counts = sorted({r["dataset"]["num_samples"] for r in results})
    return (
        '<div class="facts">'
        f"<span>model <b>{html.escape(first['model'].split('/')[-1])}</b></span>"
        f"<span>ISL <b>{first['dataset']['target_isl']}</b></span>"
        f"<span>OSL <b>{first['dataset']['target_osl']}</b></span>"
        f"<span>prompts <b>{' / '.join(str(c) for c in counts)}</b></span>"
        f"<span>decoding <b>greedy, forced length</b></span>"
        f"<span>last run <b>{max(r['timestamp'] for r in results)}</b></span>"
        "</div>"
    )


def as_html(results: list[dict], title: str = "liteinfer benchmarks") -> str:
    if not results:
        body = "<p>No results yet. Run <code>bench run --all</code>.</p>"
    else:
        sections = [_facts(results)]
        for (mode, model, isl, osl), members in sorted(group(results).items()):
            sections.append(
                f"<h2>{mode} · {html.escape(model.split('/')[-1])} · ISL={isl} OSL={osl}</h2>"
                f'<p class="meta">{MODE_NOTE[mode]} '
                f"{members[0]['dataset']['num_samples']} prompts, forced output length {osl}. "
                f"Prompt set {members[0]['dataset']['sha256'][:12]}.</p>"
            )
            if _mismatched(members):
                sections.append('<p class="warn">These runs did not all use the same prompts.</p>')
            sections.append(_table(members, mode))
        body = "".join(sections)

    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        '<p class="lede">Each config is scored against the one it improves on '
        "(<b>vs base</b>), against the start of its lineage (<b>vs first</b>), and "
        "against vLLM at the same batch width (<b>vs vLLM</b>). Above 1.00x is better. "
        "Each row names the config it improves on; vLLM rows are references, not competitors. "
        "Bars show standing on the headline metric of each table.</p>"
        f"{body}"
        "<footer>Run-to-run variance on this hardware is roughly ±4%. "
        "Generated by <code>bench report</code>.</footer>"
        "</main></body></html>"
    )
