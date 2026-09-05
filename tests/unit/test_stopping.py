"""Stop-condition behaviour, and the guarantee that both engines share it."""

from __future__ import annotations

from unittest.mock import MagicMock

from liteinfer.engine.async_llm_engine import AsyncLLMEngine
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.engine.stopping import resolve_stop_status
from liteinfer.sampling.params import SamplingParams
from liteinfer.tokenizer import IncrementalDetokenizer

EOS_ID = 2
STOP_ID = 99
MAX_MODEL_LEN = 2048


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_token_ids = {EOS_ID}
    return tokenizer


def _seq(num_output: int, params: SamplingParams, output_text: str = "") -> Sequence:
    """A sequence whose text is already decoded — which is how the engine calls the rule."""
    return Sequence(
        request_id="r0",
        prompt="hi",
        prompt_token_ids=[1],
        output_token_ids=list(range(10, 10 + num_output)),
        sampling_params=params,
        detokenizer=IncrementalDetokenizer(text=output_text),
    )


def _resolve(num_output: int, token_id: int, params: SamplingParams, output_text: str = "") -> object:
    return resolve_stop_status(
        _seq(num_output, params, output_text), token_id, _tokenizer(), MAX_MODEL_LEN
    )


def test_eos_stops_by_default() -> None:
    assert _resolve(2, EOS_ID, SamplingParams(max_tokens=64)) is SequenceStatus.FINISHED_STOPPED


def test_ignore_eos_keeps_generating_past_eos() -> None:
    assert _resolve(2, EOS_ID, SamplingParams(max_tokens=64, ignore_eos=True)) is None


def test_min_tokens_suppresses_eos_below_threshold() -> None:
    assert _resolve(2, EOS_ID, SamplingParams(max_tokens=64, min_tokens=5)) is None


def test_eos_fires_once_min_tokens_is_reached() -> None:
    status = _resolve(5, EOS_ID, SamplingParams(max_tokens=64, min_tokens=5))
    assert status is SequenceStatus.FINISHED_STOPPED


def test_min_tokens_suppresses_stop_token_below_threshold() -> None:
    params = SamplingParams(max_tokens=64, min_tokens=3, stop_token_ids=[STOP_ID])
    assert _resolve(1, STOP_ID, params) is None


def test_stop_token_fires_once_min_tokens_is_reached() -> None:
    params = SamplingParams(max_tokens=64, min_tokens=3, stop_token_ids=[STOP_ID])
    assert _resolve(3, STOP_ID, params) is SequenceStatus.FINISHED_STOPPED


def test_stop_string_stops_when_present_in_output() -> None:
    params = SamplingParams(max_tokens=64, stop=["DONE"])
    assert _resolve(2, 42, params, output_text="all DONE") is SequenceStatus.FINISHED_STOPPED


def test_max_tokens_stops_even_with_ignore_eos() -> None:
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    assert _resolve(2, 42, params) is SequenceStatus.FINISHED_LENGTH


def test_max_tokens_stops_even_below_min_tokens() -> None:
    # min_tokens can never hold a sequence past its length limit.
    params = SamplingParams(max_tokens=4, min_tokens=4)
    assert _resolve(4, 42, params) is SequenceStatus.FINISHED_LENGTH


def test_max_model_len_stops_the_sequence() -> None:
    seq = _seq(2, SamplingParams(max_tokens=999))
    assert (
        resolve_stop_status(seq, 42, _tokenizer(), max_model_len=3)
        is SequenceStatus.FINISHED_LENGTH
    )


def test_engine_honours_ignore_eos() -> None:
    # Regression guard: the engine once carried its own copy of this rule and
    # silently ignored ignore_eos/min_tokens.
    engine = MagicMock()
    engine.tokenizer = _tokenizer()
    engine.config.max_model_len = MAX_MODEL_LEN
    seq = _seq(2, SamplingParams(max_tokens=64, ignore_eos=True))

    AsyncLLMEngine._maybe_finish(engine, seq, EOS_ID)

    assert seq.status is SequenceStatus.WAITING
