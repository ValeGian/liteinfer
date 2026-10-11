# pyright: reportPrivateImportUsage=false
"""A step holding prompts and sampled tokens together, on CPU.

A mixed step is one packed forward here too, read through the dense loop; what
must hold is that its caller cannot tell it from separate passes. The paged
kernel's version needs CUDA and is checked in `test_mixed_step_gpu.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from liteinfer import AsyncLLM
from liteinfer.cache.block_pool import BlockPoolExhaustedError
from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.sequence import Sequence
from liteinfer.outputs import RequestOutput
from liteinfer.sampling.params import SamplingParams
from tests.integration.sequences import running_sequence

# One forward and three run their GEMMs over different token counts, and CPU
# BLAS blocks the reduction differently per shape, so fp32 sums round
# differently: observed 1.5e-5 on logits near 74. A row out of place differs at
# the scale of the logits themselves.
_TOLERANCE = {"rtol": 0, "atol": 1e-4}


def _runner(model_dir: Path) -> ContinuousModelRunner:
    runner = ContinuousModelRunner(
        EngineConfig(
            model=str(model_dir), device="cpu", dtype=torch.float32,  # type: ignore[arg-type]
            max_num_seqs=4, max_model_len=64, block_size=4,
        )
    )
    runner.load_model()
    return runner


def _decoding(runner: ContinuousModelRunner) -> Sequence:
    """A sequence whose prompt is cached and whose sampled token is not yet."""
    seq = running_sequence("decoding", prompt_len=11, offset=0)
    runner.execute([seq])
    seq.output_token_ids.append(7)
    return seq


def test_the_next_chunk_of_a_decoding_sequence_is_its_last_sampled_token(tiny_llama_dir: Path):
    runner = _runner(tiny_llama_dir)
    seq = _decoding(runner)

    assert runner._next_chunks([seq], None)[0].token_ids == [7]


def test_a_chunk_crossing_from_prompt_into_output_is_refused(tiny_llama_dir: Path):
    """A sequence still computing its prompt has sampled nothing, so this is a caller bug."""
    runner = _runner(tiny_llama_dir)
    seq = running_sequence("crossing", prompt_len=5, offset=0)
    seq.output_token_ids.append(7)
    assert runner._cache is not None, "load_model() builds the cache"
    runner._cache.register(seq.request_id)

    with pytest.raises(ValueError, match="crosses the end"):
        runner._next_chunks([seq], None)


def test_a_mixed_step_answers_each_sequence_in_the_order_it_was_given(tiny_llama_dir: Path):
    """One pass, rows in the given order: a row out of place would hand a sequence another's logits."""
    mixed_runner, separate_runner = _runner(tiny_llama_dir), _runner(tiny_llama_dir)
    mixed = mixed_runner.execute(
        [running_sequence("prompt-a", 9, 30), _decoding(mixed_runner), running_sequence("prompt-b", 6, 60)]
    )
    separate = torch.cat([
        separate_runner.execute([running_sequence("prompt-a", 9, 30)]),
        separate_runner.execute([_decoding(separate_runner)]),
        separate_runner.execute([running_sequence("prompt-b", 6, 60)]),
    ])

    torch.testing.assert_close(mixed, separate, **_TOLERANCE)


# --- a step the pool cannot hold ---------------------------------------------

# One block of 16 tokens: the running sequence fits in it for its whole
# generation, and the late prompt needs a second block that is never there.
_POOL = {"num_gpu_blocks": 1, "block_size": 16}
_RUNNING_TOKENS = 8


class _LatePrompt(NamedTuple):
    running_tokens: list[int]
    """What the running sequence generated in the end."""
    running_tokens_when_refused: int
    """How far it had got when the late prompt failed, which shows the two overlapped."""
    late_error: BaseException | None


def _late_prompt_on_a_full_pool(model_dir: Path) -> _LatePrompt:
    """Start one sequence, admit a prompt that cannot fit while it runs, and see what each got."""

    async def run() -> _LatePrompt:
        llm = AsyncLLM(
            str(model_dir), device="cpu", dtype=torch.float32,  # type: ignore[arg-type]
            max_num_seqs=2, max_model_len=64, **_POOL,
        )
        async with llm:
            params = SamplingParams(max_tokens=_RUNNING_TOKENS, temperature=0.0, ignore_eos=True)
            tokens: list[int] = []
            tokens_when_refused = -1
            late: asyncio.Task | None = None
            async for event in llm.stream("tok2 tok3 tok4", params):
                tokens = list(event.output_token_ids)
                if late is None:
                    late_prompt = " ".join(f"tok{5 + i}" for i in range(20))
                    late = asyncio.ensure_future(llm.generate(late_prompt, params))
                elif late.done() and tokens_when_refused < 0:
                    tokens_when_refused = len(tokens) - 1
            assert late is not None
            (late_result,) = await asyncio.gather(late, return_exceptions=True)
            error = late_result if isinstance(late_result, BaseException) else None
            return _LatePrompt(tokens, tokens_when_refused, error)

    return asyncio.run(run())


def test_a_prompt_the_pool_cannot_hold_fails_with_the_pool_error(tiny_llama_dir: Path):
    assert isinstance(_late_prompt_on_a_full_pool(tiny_llama_dir).late_error, BlockPoolExhaustedError)


def test_the_refused_prompt_arrived_while_the_running_sequence_was_generating(tiny_llama_dir: Path):
    """Keeps the test below honest: refused before the first token or after the last proves nothing."""
    refused_at = _late_prompt_on_a_full_pool(tiny_llama_dir).running_tokens_when_refused

    assert 0 < refused_at < _RUNNING_TOKENS


def test_a_prompt_the_pool_cannot_hold_does_not_stop_a_running_sequence(tiny_llama_dir: Path):
    assert len(_late_prompt_on_a_full_pool(tiny_llama_dir).running_tokens) == _RUNNING_TOKENS


def _generate_on_a_small_pool(
    model_dir: Path, prompt_lens: list[int], max_tokens: int, **pool
) -> tuple[RequestOutput | BaseException, ...]:
    """Every prompt submitted at once; each one's output, or the error that ended it."""

    async def run():
        llm = AsyncLLM(
            str(model_dir), device="cpu", dtype=torch.float32,  # type: ignore[arg-type]
            max_num_seqs=len(prompt_lens), max_model_len=64, block_size=16, **pool,
        )
        async with llm:
            params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
            prompts = [" ".join(f"tok{2 + 7 * i + j}" for j in range(n)) for i, n in enumerate(prompt_lens)]
            requests = [llm.generate(prompt, params) for prompt in prompts]
            results = await asyncio.gather(*requests, return_exceptions=True)
            return tuple(r if isinstance(r, BaseException) else r[0] for r in results)

    return asyncio.run(run())


def test_a_full_pool_sheds_the_newcomer_rather_than_a_prompt_part_way_through(tiny_llama_dir: Path):
    """The part-way prompt fits on its own; dropping it too would discard work for nothing.

    Two 19-token prompts, an 8-token budget, two blocks: the first is chunked
    8 / 8 / 3, and its last chunk needs its second block in the step that admits
    the second prompt, which needs one too.
    """
    first, _ = _generate_on_a_small_pool(
        tiny_llama_dir, [19, 19], max_tokens=4, num_gpu_blocks=2, max_num_batched_tokens=8
    )

    assert isinstance(first, RequestOutput)


def test_a_full_pool_sheds_a_part_way_prompt_rather_than_a_decoding_sequence(tiny_llama_dir: Path):
    """No newcomer to drop: a 3-token prompt decodes while a 19-token one is chunked
    5 / 7 / 7 beside it, and the last chunk needs a block that is not there."""
    decoding, _ = _generate_on_a_small_pool(
        tiny_llama_dir, [3, 19], max_tokens=4, num_gpu_blocks=2, max_num_batched_tokens=8
    )

    assert isinstance(decoding, RequestOutput)


def test_a_full_pool_with_only_decodes_left_fails_them(tiny_llama_dir: Path):
    """Nothing smaller to give up: one sequence outgrows the only block there is."""
    (only,) = _generate_on_a_small_pool(tiny_llama_dir, [3], max_tokens=20, num_gpu_blocks=1)

    assert isinstance(only, BlockPoolExhaustedError)
