"""Unit tests for LLMEngine._maybe_finish stop logic.

Tests ignore_eos and min_tokens by calling _maybe_finish directly via an
unbound-method call on a MagicMock that satisfies the attribute lookups.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from liteinfer.engine.llm_engine import LLMEngine
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams

_EOS_ID = 2


def _engine(eos_id: int = _EOS_ID, max_model_len: int = 2048) -> MagicMock:
    eng = MagicMock()
    eng.tokenizer.eos_token_ids = {eos_id}
    eng.config.max_model_len = max_model_len
    return eng


def _seq(
    output_token_ids: list[int],
    params: SamplingParams,
) -> Sequence:
    return Sequence(
        request_id="r0",
        prompt="hi",
        prompt_token_ids=[1],
        output_token_ids=output_token_ids,
        sampling_params=params,
    )


def _finish(eng, seq, token_id: int) -> None:
    LLMEngine._maybe_finish(eng, seq, token_id)


def test_eos_stops_by_default() -> None:
    eng = _engine()
    seq = _seq([10, 20], SamplingParams(max_tokens=64))
    _finish(eng, seq, _EOS_ID)
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_ignore_eos_skips_eos_stop() -> None:
    eng = _engine()
    seq = _seq([10, 20], SamplingParams(max_tokens=64, ignore_eos=True))
    _finish(eng, seq, _EOS_ID)
    assert seq.status == SequenceStatus.WAITING


def test_max_tokens_stops_regardless_of_ignore_eos() -> None:
    eng = _engine()
    seq = _seq([10, 20], SamplingParams(max_tokens=2, ignore_eos=True))
    _finish(eng, seq, 99)  # non-EOS token
    assert seq.status == SequenceStatus.FINISHED_LENGTH


def test_min_tokens_suppresses_eos_until_threshold() -> None:
    eng = _engine()
    # min_tokens=5, only 2 output tokens so far → EOS should not stop
    seq = _seq([10, 20], SamplingParams(max_tokens=64, min_tokens=5))
    _finish(eng, seq, _EOS_ID)
    assert seq.status == SequenceStatus.WAITING


def test_min_tokens_allows_eos_after_threshold() -> None:
    eng = _engine()
    # min_tokens=2, already 2 output tokens → EOS should stop
    seq = _seq([10, 20], SamplingParams(max_tokens=64, min_tokens=2))
    _finish(eng, seq, _EOS_ID)
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_min_tokens_suppresses_stop_token_ids_until_threshold() -> None:
    eng = _engine()
    stop_id = 99
    seq = _seq([10], SamplingParams(max_tokens=64, min_tokens=3, stop_token_ids=[stop_id]))
    _finish(eng, seq, stop_id)
    assert seq.status == SequenceStatus.WAITING


def test_stop_token_id_fires_after_min_tokens() -> None:
    eng = _engine()
    stop_id = 99
    seq = _seq([10, 20, 30], SamplingParams(max_tokens=64, min_tokens=3, stop_token_ids=[stop_id]))
    _finish(eng, seq, stop_id)
    assert seq.status == SequenceStatus.FINISHED_STOPPED
