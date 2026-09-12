"""Report grouping and the three comparison scores."""

from __future__ import annotations

from benchmarks import report


def _result(config: str, mode: str, summary: dict, **overrides) -> dict:
    record = {
        "config": config,
        "engine": "vllm" if config.startswith("vllm") else "liteinfer",
        "description": "",
        "baseline": None,
        "mode": mode,
        "model": "test/model",
        "max_num_seqs": 1,
        "timestamp": "2026-08-29T12:00:00+00:00",
        "revision": "aaaaaaa",
        "dataset": {"target_isl": 16, "target_osl": 8, "num_samples": 3, "sha256": "abc"},
        "summary": summary,
    }
    record["dataset"].update(overrides.pop("dataset", {}))
    return record | overrides


def _throughput(config: str, tok_s: float, **overrides) -> dict:
    summary = {"output_tokens_per_s": tok_s, "requests_per_s": 1.0, "wall_time_s": 1.0}
    return _result(config, "throughput", summary, **overrides)


def _latency(config: str, itl_ms: float, **overrides) -> dict:
    summary = {
        "itl_p50_ms": itl_ms,
        "itl_p95_ms": itl_ms,
        "ttft_p50_ms": 1.0,
        "ttft_p95_ms": 1.0,
        "e2e_p50_ms": 1.0,
    }
    return _result(config, "latency", summary, **overrides)


def _score(members: list[dict], mode: str, config: str) -> report.Row:
    return next(r for r in report.rows(members, mode) if r.result["config"] == config)


# --- grouping ---------------------------------------------------------------


def test_results_split_by_mode() -> None:
    assert len(report.group([_throughput("a", 10.0), _latency("a", 10.0)])) == 2


def test_results_split_by_benchmark_shape() -> None:
    other = _throughput("a", 10.0, dataset={"target_osl": 64})
    assert len(report.group([_throughput("a", 10.0), other])) == 2


def test_two_mixed_runs_with_different_caps_are_different_shapes() -> None:
    """Different caps hold different prompts, so a ratio across them compares workloads."""
    narrow = _throughput("a", 10.0, dataset={"target_isl": None, "max_isl": 512})
    wide = _throughput("a", 10.0, dataset={"target_isl": None, "max_isl": 2048})

    assert len(report.group([narrow, wide])) == 2


def test_a_result_written_before_mixed_shapes_still_groups() -> None:
    """Stored results carry no `max_isl`, and a fixed ISL has no cap to carry."""
    stored = _throughput("a", 10.0)

    assert "max_isl" not in stored["dataset"] and report.shape_of(stored) == (16, 8, None)


def test_a_mixed_shape_is_labelled_by_its_cap() -> None:
    """A reader must not mistake a mixed column for a prompt length."""
    assert report._shape_label((None, 128, 2048)) == "mix2048/128"


def test_mixed_shapes_sort_after_fixed_ones() -> None:
    """Mixed is a different workload, not the long end of a trend."""
    shapes = [(None, 128, 2048), (3584, 128, None), (128, 256, None)]

    assert sorted(shapes, key=report._shape_order)[-1] == (None, 128, 2048)


# --- vs base ----------------------------------------------------------------


def test_higher_throughput_than_baseline_scores_above_one() -> None:
    members = [
        _throughput("liteinfer-nocache", 100.0),
        _throughput("liteinfer-eager", 200.0, baseline="liteinfer-nocache"),
    ]
    assert _score(members, "throughput", "liteinfer-eager").vs_base == 2.0


def test_lower_latency_than_baseline_scores_above_one() -> None:
    members = [
        _latency("liteinfer-nocache", 100.0),
        _latency("liteinfer-eager", 50.0, baseline="liteinfer-nocache"),
    ]
    assert _score(members, "latency", "liteinfer-eager").vs_base == 2.0


def test_a_config_without_a_baseline_has_no_delta() -> None:
    members = [_throughput("liteinfer-nocache", 100.0)]
    assert _score(members, "throughput", "liteinfer-nocache").vs_base is None


def test_a_baseline_missing_from_the_group_has_no_delta() -> None:
    members = [_throughput("liteinfer-eager", 200.0, baseline="liteinfer-nocache")]
    assert _score(members, "throughput", "liteinfer-eager").vs_base is None


# --- vs first (cumulative) --------------------------------------------------


def test_cumulative_compares_against_the_root_of_the_lineage() -> None:
    members = [
        _throughput("liteinfer-nocache", 100.0),
        _throughput("liteinfer-eager", 200.0, baseline="liteinfer-nocache"),
        _throughput("liteinfer-native-eager", 400.0, baseline="liteinfer-eager"),
    ]
    assert _score(members, "throughput", "liteinfer-native-eager").vs_root == 4.0


def test_the_root_config_has_no_cumulative_score() -> None:
    members = [_throughput("liteinfer-nocache", 100.0)]
    assert _score(members, "throughput", "liteinfer-nocache").vs_root is None


# --- vs vLLM ----------------------------------------------------------------


def test_vllm_comparison_matches_on_batch_width() -> None:
    members = [
        _throughput("liteinfer-paged-b4", 100.0, max_num_seqs=4),
        _throughput("vllm", 999.0, max_num_seqs=1),
        _throughput("vllm-b4", 400.0, max_num_seqs=4),
    ]
    assert _score(members, "throughput", "liteinfer-paged-b4").vs_vllm == 0.25


def test_no_vllm_at_that_width_leaves_the_comparison_empty() -> None:
    members = [
        _throughput("liteinfer-paged-b4", 100.0, max_num_seqs=4),
        _throughput("vllm", 999.0, max_num_seqs=1),
    ]
    assert _score(members, "throughput", "liteinfer-paged-b4").vs_vllm is None


def test_vllm_is_not_compared_against_itself() -> None:
    members = [_throughput("vllm", 999.0, max_num_seqs=1)]
    assert _score(members, "throughput", "vllm").vs_vllm is None


# --- ordering and rendering -------------------------------------------------


def test_rows_follow_the_declared_lineage_order() -> None:
    out_of_order = [_throughput("liteinfer-paged", 1.0), _throughput("liteinfer-nocache", 1.0)]
    ordered = [r.result["config"] for r in report.rows(out_of_order, "throughput")]
    assert ordered == ["liteinfer-nocache", "liteinfer-paged"]


def test_each_row_names_the_config_it_improves_on() -> None:
    members = [_throughput("liteinfer-eager", 1.0, baseline="liteinfer-nocache")]
    assert _score(members, "throughput", "liteinfer-eager").base == "liteinfer-nocache"


# --- what a delta is a claim about ------------------------------------------


def test_a_delta_against_a_removed_config_is_marked_in_text() -> None:
    """`liteinfer-paged-b4` is removed, so its stored number is frozen at that engine.

    A delta against it therefore compares two engines, not two configs, and the
    reader has to be able to tell that from the delta that compares two runnable
    configs measured together.
    """
    members = [_throughput("liteinfer-continuous", 10.0, baseline="liteinfer-paged-b4")]

    assert "liteinfer-paged-b4*" in report.as_text(members)


def test_a_delta_against_a_runnable_config_is_not_marked_in_text() -> None:
    members = [_throughput("liteinfer-sdpa", 10.0, baseline="liteinfer-continuous")]

    assert "liteinfer-continuous*" not in report.as_text(members)


def test_a_delta_against_a_removed_config_is_explained_in_html() -> None:
    members = [_throughput("liteinfer-continuous", 10.0, baseline="liteinfer-paged-b4")]

    assert "spans two engines" in report.as_html(members)


def test_differing_prompt_sets_are_flagged_in_text() -> None:
    mismatched = _throughput("liteinfer-eager", 10.0, dataset={"sha256": "different"})
    assert "WARNING" in report.as_text([_throughput("liteinfer-nocache", 10.0), mismatched])


def test_html_is_standalone() -> None:
    page = report.as_html([_throughput("liteinfer-nocache", 10.0)])
    assert "http://" not in page and "https://" not in page


def test_each_section_states_its_own_prompt_set_digest() -> None:
    # The digest covers the trimmed sample set, so modes run at different sample
    # counts carry different digests and each section must show its own.
    latency = _latency("liteinfer-nocache", 10.0, dataset={"num_samples": 50, "sha256": "fifty"})
    page = report.as_html([_throughput("liteinfer-nocache", 10.0), latency])
    assert "fifty" in page and "abc" in page


def test_html_reports_an_empty_results_dir() -> None:
    assert "No results yet" in report.as_html([])


def test_text_reports_an_empty_results_dir() -> None:
    assert report.as_text([]) == "No results found."


def test_shapes_are_collected_across_results() -> None:
    long_prompt = _throughput("liteinfer-continuous", 900.0, dataset={"target_isl": 1024})
    shapes, _ = report.by_shape([_throughput("liteinfer-continuous", 1200.0), long_prompt], "throughput")
    assert shapes == [(16, 8, None), (1024, 8, None)]


def test_each_config_reports_its_value_per_shape() -> None:
    long_prompt = _throughput("liteinfer-continuous", 900.0, dataset={"target_isl": 1024})
    _, values = report.by_shape([_throughput("liteinfer-continuous", 1200.0), long_prompt], "throughput")
    assert values["liteinfer-continuous"] == {(16, 8, None): 1200.0, (1024, 8, None): 900.0}


def test_a_single_shape_produces_no_trend_section() -> None:
    page = report.as_html([_throughput("liteinfer-continuous", 1200.0)])
    assert "across shapes" not in page


def test_two_shapes_produce_a_trend_section() -> None:
    long_prompt = _throughput("liteinfer-continuous", 900.0, dataset={"target_isl": 1024})
    page = report.as_html([_throughput("liteinfer-continuous", 1200.0), long_prompt])
    assert "across shapes" in page


# ---------------------------------------------------------------------------
# Whether a ratio compares two runs that are comparable at all (§8.4)
# ---------------------------------------------------------------------------


def _pair(**overrides) -> list[dict]:
    """A config and the baseline it improves on, identical but for `overrides`."""
    base = _throughput("liteinfer-nocache", 10.0, revision="aaaaaaa")
    improved = _throughput(
        "liteinfer-eager", 20.0, baseline="liteinfer-nocache", revision="aaaaaaa"
    )
    improved.update({k: v for k, v in overrides.items() if k != "dataset"})
    if "dataset" in overrides:
        improved["dataset"] = improved["dataset"] | overrides["dataset"]
    return [base, improved]


def test_a_delta_between_two_runs_of_the_same_engine_is_trusted() -> None:
    assert _score(_pair(), "throughput", "liteinfer-eager").unsound == {}


def test_a_delta_across_two_revisions_is_not_trusted() -> None:
    """An ungated change lands in every config at once, so a baseline measured
    before it makes the newer row look better than the change was worth."""
    row = _score(_pair(revision="bbbbbbb"), "throughput", "liteinfer-eager")

    assert row.unsound["base"] == "different revisions"


def test_a_delta_between_two_runs_of_one_dirty_revision_is_still_trusted() -> None:
    """A sweep writes a result per run, so the tree is dirty for every run after
    the first. Marking that would flag every ratio in the report; the weaker
    evidence is said once under the table instead."""
    pair = _pair()
    for result in pair:
        result["revision"] = "aaaaaaa-dirty"

    assert _score(pair, "throughput", "liteinfer-eager").unsound == {}


def test_a_dirty_revision_is_noted_under_the_table() -> None:
    pair = _pair()
    for result in pair:
        result["revision"] = "aaaaaaa-dirty"

    assert "uncommitted tree" in report.as_text(pair)


def test_a_delta_between_a_clean_and_a_dirty_revision_is_not_trusted() -> None:
    """Different strings mean different code, however they differ."""
    row = _score(_pair(revision="aaaaaaa-dirty"), "throughput", "liteinfer-eager")

    assert row.unsound["base"] == "different revisions"


def test_a_cross_engine_ratio_ignores_our_revision() -> None:
    """A vLLM row's revision is a fact about vLLM, so comparing the two says nothing."""
    ours = _throughput("liteinfer-eager", 20.0, revision="aaaaaaa")
    theirs = _throughput("vllm", 10.0, revision="zzzzzzz")

    assert _score([ours, theirs], "throughput", "liteinfer-eager").unsound == {}


def test_a_delta_across_two_prompt_sets_is_not_trusted() -> None:
    """`benchmarks/datasets/` is gitignored, so a regenerated file takes a new digest."""
    row = _score(_pair(dataset={"sha256": "different"}), "throughput", "liteinfer-eager")

    assert row.unsound["base"] == "different prompts"


def test_results_predating_the_revision_field_fall_back_to_the_clock() -> None:
    """Older stored results carry no revision; the clock still catches the worst."""
    base = _throughput("liteinfer-nocache", 10.0, timestamp="2026-01-01T00:00:00+00:00")
    improved = _throughput("liteinfer-eager", 20.0, baseline="liteinfer-nocache")
    del base["revision"], improved["revision"]

    assert "apart" in _score([base, improved], "throughput", "liteinfer-eager").unsound["base"]


def test_a_config_with_no_baseline_has_nothing_to_distrust() -> None:
    """There is no ratio, so there is no doubt to record."""
    assert _score(_pair(), "throughput", "liteinfer-nocache").unsound == {}


def test_an_untrustworthy_ratio_is_marked_in_the_text_report() -> None:
    """The doubt has to travel with the number, not sit in a footnote."""
    assert "2.00x~" in report.as_text(_pair(revision="bbbbbbb"))


def test_an_untrustworthy_ratio_says_why_in_the_html_report() -> None:
    assert "different revisions" in report.as_html(_pair(revision="bbbbbbb"))


def test_a_trustworthy_ratio_is_left_unmarked_in_the_text_report() -> None:
    assert "2.00x~" not in report.as_text(_pair())
