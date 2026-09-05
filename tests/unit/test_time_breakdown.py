"""Time attribution: the loop must account for where its wall time went."""

from __future__ import annotations

import pytest

from liteinfer.engine.metrics import TimeBreakdown


def test_shares_are_empty_before_the_first_step():
    assert TimeBreakdown().shares() == {}


def test_unattributed_is_what_the_stages_do_not_explain():
    breakdown = TimeBreakdown(forward=6.0, sample=1.0, deliver=1.0, schedule=0.5, loop=10.0)

    assert breakdown.unattributed == pytest.approx(1.5)


def test_unattributed_never_goes_negative():
    """Stages are timed independently, so rounding must not produce a negative."""
    breakdown = TimeBreakdown(forward=10.001, loop=10.0)

    assert breakdown.unattributed == 0.0


def test_shares_sum_to_one():
    breakdown = TimeBreakdown(forward=6.0, sample=1.0, deliver=2.0, schedule=0.5, loop=10.0)

    assert sum(breakdown.shares().values()) == pytest.approx(1.0)


def test_forward_share_is_the_fraction_of_loop_time():
    breakdown = TimeBreakdown(forward=7.5, loop=10.0)

    assert breakdown.shares()["forward"] == pytest.approx(0.75)


def test_add_accumulates_across_steps():
    breakdown = TimeBreakdown()

    breakdown.add("forward", 1.5)
    breakdown.add("forward", 2.5)

    assert breakdown.forward == pytest.approx(4.0)
