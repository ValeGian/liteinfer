"""AsyncLLMEngine — continuous-batching inference loop driven by asyncio.

Design
------
A single background asyncio Task runs the generate loop. It processes engine
steps synchronously (GPU-bound forward passes are not async-friendly) and
yields control to the event loop between steps via ``await asyncio.sleep(0)``.
This lets concurrent coroutines (e.g., multiple ``generate_stream`` callers)
submit new requests and read from their output queues without being blocked.

Per-request delivery
    Each request is assigned an ``asyncio.Queue[StreamEvent | None]``. The
    engine loop pushes a ``StreamEvent`` after every step in which the request
    produces a token, then pushes ``None`` as a sentinel when the sequence
    finishes. Consumers iterate with ``async for`` until they see ``None``.

Step structure
    1. ``remove_finished`` — evict individually-done sequences, free KV blocks.
    2. ``schedule`` — fill empty slots with waiting sequences.
    3. ``prefill`` newly admitted sequences (separate forward pass).
    4. ``decode`` already-running sequences (separate forward pass).
    5. Deliver ``StreamEvent`` objects to per-request queues.

The two-pass step (prefill + decode as separate forward calls) keeps the
implementation simple at the cost of an extra kernel launch when new sequences
join a running decode batch. See roadmap §1.3 for the planned single-pass
chunked-prefill upgrade that eliminates this overhead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.continuous_scheduler import ContinuousScheduler
from liteinfer.engine.metrics import (
    EngineStats,
    Phase,
    StepMetrics,
    StepTimer,
    peak_gpu_memory_bytes,
)
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.engine.stopping import resolve_stop_status
from liteinfer.outputs import StreamEvent
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler
from liteinfer.tokenizer import Tokenizer

# A request's stream carries events, then either None (done) or the error that
# ended it.
_RequestQueue = asyncio.Queue[StreamEvent | Exception | None]

_IDLE_POLL_S = 0.05

_FINISH_REASONS: dict[SequenceStatus, str] = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
}


class AsyncLLMEngine:
    """Continuous-batching inference engine backed by an asyncio event loop."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.scheduler = ContinuousScheduler(config)
        self.model_runner = ContinuousModelRunner(config)
        self.sampler = Sampler()
        self._step_idx = 0
        self.stats = EngineStats()

        self._request_queues: dict[str, _RequestQueue] = {}
        self._pending: asyncio.Queue = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    @property
    def tokenizer(self) -> Tokenizer:
        return self.model_runner.tokenizer

    async def start(self) -> None:
        """Load model and start the background generate loop."""
        self.model_runner.load_model()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await its completion."""
        self._shutdown.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

    async def generate_stream(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Submit a request and stream ``StreamEvent`` objects until completion.

        A failure belonging to this request is re-raised here rather than
        silently ending the stream.
        """
        if self._loop_task is None or self._loop_task.done():
            raise RuntimeError("engine loop is not running; call start() first")

        queue: _RequestQueue = asyncio.Queue()
        self._request_queues[request_id] = queue
        await self._pending.put((request_id, prompt, sampling_params))

        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._drain_pending()
                if self.scheduler.has_unfinished():
                    self._step()
                else:
                    try:
                        item = await asyncio.wait_for(self._pending.get(), timeout=_IDLE_POLL_S)
                        self._admit(item)
                    except asyncio.TimeoutError:
                        pass
                await asyncio.sleep(0)
        except BaseException as error:
            # The loop is the only thing that ever completes a request, so if it
            # dies every waiter must hear about it instead of hanging.
            self._fail_all(error if isinstance(error, Exception) else RuntimeError(repr(error)))
            raise

    def _drain_pending(self) -> None:
        while not self._pending.empty():
            self._admit(self._pending.get_nowait())

    def _admit(self, item: tuple[str, str, SamplingParams]) -> None:
        """Tokenize and queue one request. A bad request fails only itself."""
        request_id, prompt, sampling_params = item
        try:
            self._enqueue(request_id, prompt, sampling_params)
        except Exception as error:
            self._fail(request_id, error)

    def _enqueue(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> None:
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens, >= max_model_len={self.config.max_model_len}")
        seq = Sequence(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=list(token_ids),
            sampling_params=sampling_params,
        )
        self.scheduler.add(seq)

    def _step(self) -> None:
        finished = self.scheduler.remove_finished()
        for seq in finished:
            self.model_runner.deregister_sequence(seq)
            queue = self._request_queues.pop(seq.request_id, None)
            if queue is not None:
                queue.put_nowait(None)

        sched = self.scheduler.schedule()
        if not sched.prefill_seqs and not sched.decode_seqs:
            return

        for phase, seqs in ((Phase.PREFILL, sched.prefill_seqs), (Phase.DECODE, sched.decode_seqs)):
            if not seqs:
                continue
            try:
                self._forward(phase, seqs)
            except Exception as error:
                self._abort(seqs, error)  # the pass failed, so its sequences cannot continue
                return

        newly_finished = 0
        for seq in sched.all_seqs:
            queue = self._request_queues.get(seq.request_id)
            if queue is not None:
                queue.put_nowait(self._build_event(seq))
            if seq.is_finished:
                newly_finished += 1

        self.stats.num_requests_finished += newly_finished

    def _fail(self, request_id: str, error: Exception) -> None:
        """Hand `error` to one waiting caller and forget the request."""
        queue = self._request_queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(error)

    def _fail_all(self, error: Exception) -> None:
        for request_id in list(self._request_queues):
            self._fail(request_id, error)

    def _abort(self, seqs: list[Sequence], error: Exception) -> None:
        for seq in seqs:
            seq.status = SequenceStatus.FINISHED_ABORTED
            self._fail(seq.request_id, error)

    def _forward(self, phase: Phase, seqs: list[Sequence]) -> None:
        """Run one forward pass and record it.

        Prefill and decode are separate passes, so a step that admits new
        sequences records two — which is what makes the two-pass cost (§1.3)
        visible in `stats`.
        """
        run = self.model_runner.prefill if phase is Phase.PREFILL else self.model_runner.decode
        input_tokens = (
            sum(len(seq.prompt_token_ids) for seq in seqs) if phase is Phase.PREFILL else len(seqs)
        )
        with StepTimer(self.model_runner.device) as timer:
            logits = run(seqs)
            sampled = self.sampler(logits, [seq.sampling_params for seq in seqs])
            self._apply_sampled(seqs, sampled)

        if not self.config.collect_stats:
            return
        self.stats.record(
            StepMetrics(
                step_idx=self._step_idx,
                phase=phase,
                num_seqs=len(seqs),
                input_tokens=input_tokens,
                new_tokens=len(seqs),
                wall_time_s=timer.elapsed,
                peak_gpu_mem_bytes=peak_gpu_memory_bytes(self.model_runner.device),
            )
        )
        self._step_idx += 1

    def _apply_sampled(self, seqs: list[Sequence], sampled) -> None:
        for i, seq in enumerate(seqs):
            token_id = int(sampled[i].item())
            seq.output_token_ids.append(token_id)
            self._maybe_finish(seq, token_id)

    def _build_event(self, seq: Sequence) -> StreamEvent:
        text = self.tokenizer.decode(seq.output_token_ids)
        return StreamEvent(
            request_id=seq.request_id,
            prompt=seq.prompt,
            output_token_ids=list(seq.output_token_ids),
            text=text,
            is_finished=seq.is_finished,
            finish_reason=_FINISH_REASONS.get(seq.status) if seq.is_finished else None,
        )

    def _maybe_finish(self, seq: Sequence, last_token_id: int) -> None:
        status = resolve_stop_status(seq, last_token_id, self.tokenizer, self.config.max_model_len)
        if status is not None:
            seq.status = status
