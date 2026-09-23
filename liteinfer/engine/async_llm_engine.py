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
    2. ``schedule`` — spend the step's token budget: one token per decoding
       sequence, the rest of a prompt (or as much of it as fits) per prefilling
       one, and admit waiting sequences into free slots with what is left.
    3. ``prefill`` the sequences still computing their prompt (one forward pass).
    4. ``decode`` the ones past it (a second forward pass).
    5. Deliver ``StreamEvent`` objects to per-request queues, for every sequence
       that sampled a token — a prompt chunk that stops short of the prompt's
       end has none yet.

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


class EngineOverloaded(RuntimeError):
    """The waiting queue is full. The request was not accepted; retry later.

    Its own type so a caller can tell "come back later" apart from "this
    request is malformed", which is the difference between retrying and giving
    up.
    """

_FINISH_REASONS: dict[SequenceStatus, str] = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
}


def _by_phase(seqs: list[Sequence]) -> list[tuple[Phase, list[Sequence]]]:
    """Split a step's sequences into the two passes the runner still issues, skipping empty ones.

    The scheduler has no phases — it grants tokens — so this is the one place
    they remain, and it is what §1.3 removes. A sequence still computing its
    prompt goes to prefill even when its chunk is a single token; everything
    else is decoding its last sampled token.
    """
    prefill = [seq for seq in seqs if not seq.is_prompt_computed]
    decode = [seq for seq in seqs if seq.is_prompt_computed]
    return [(phase, group) for phase, group in ((Phase.PREFILL, prefill), (Phase.DECODE, decode)) if group]


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
        # Requests whose caller wants only the completed generation. `generate`
        # iterates every event and keeps the last, so building and queueing the
        # intermediate ones is work nobody reads — 7.3% of the loop at 128
        # concurrent sequences, since it is paid per sequence per step.
        self._final_event_only: set[str] = set()
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
        stream_tokens: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Submit a request and stream ``StreamEvent`` objects until completion.

        A failure belonging to this request is re-raised here rather than
        silently ending the stream.

        `stream_tokens=False` yields one event, when the generation completes.
        It is for callers that only want the finished text — `generate` is one —
        and it exists because the events it skips are built per sequence per
        step and then discarded.
        """
        if self._loop_task is None or self._loop_task.done():
            raise RuntimeError("engine loop is not running; call start() first")
        if self.num_waiting >= self.config.max_waiting_seqs:
            raise EngineOverloaded(
                f"{self.num_waiting} requests already waiting "
                f"(max_waiting_seqs={self.config.max_waiting_seqs}); retry later"
            )

        queue: _RequestQueue = asyncio.Queue()
        self._request_queues[request_id] = queue
        if not stream_tokens:
            self._final_event_only.add(request_id)
        # Enqueued without awaiting, so the check above and this cannot interleave.
        self._pending.put_nowait((request_id, prompt, sampling_params))

        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    @property
    def num_waiting(self) -> int:
        """Requests accepted but not yet running, wherever they are queued."""
        return self._pending.qsize() + len(self.scheduler.waiting)

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
        if not token_ids:
            # Nothing to condition on and no position to read logits from. Refused
            # here, where it fails only this request: admitted, it would owe the
            # scheduler zero tokens, which is an engine invariant and not a request error.
            raise ValueError(f"prompt {prompt!r} encodes to no tokens")
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
        with self._timed("loop"):
            self._run_step()

    def _run_step(self) -> None:
        with self._timed("schedule", sync=False):
            finished = self.scheduler.remove_finished()
            for seq in finished:
                self.model_runner.deregister_sequence(seq)
                queue = self._forget(seq.request_id)
                if queue is not None:
                    queue.put_nowait(None)
            sched = self.scheduler.schedule()

        if sched.is_empty:
            return

        sampled_seqs: list[Sequence] = []
        for phase, seqs in _by_phase(sched.seqs):
            num_tokens = [sched.num_scheduled_tokens[seq.request_id] for seq in seqs]
            try:
                sampled_seqs += self._forward(phase, seqs, num_tokens)
            except Exception as error:
                self._abort(seqs, error)  # the pass failed, so its sequences cannot continue
                return

        with self._timed("deliver", sync=False):
            newly_finished = 0
            for seq in sampled_seqs:
                if seq.is_finished:
                    newly_finished += 1
                queue = self._request_queues.get(seq.request_id)
                if queue is None:
                    continue
                if seq.is_finished or seq.request_id not in self._final_event_only:
                    queue.put_nowait(self._build_event(seq))

        self.stats.num_requests_finished += newly_finished

    def _timed(self, stage: str, sync: bool = True) -> StepTimer:
        """Charge a block of the step to one stage of `stats.time`."""
        return StepTimer(
            self.model_runner.device,
            self.stats.time if self.config.collect_stats else None,
            stage,
            sync=sync,
        )

    def _forget(self, request_id: str) -> _RequestQueue | None:
        """Drop everything the engine holds for a request, returning its queue.

        One funnel, so a new piece of per-request state cannot be cleaned up on
        the completion path and leaked on the failure path.
        """
        self._final_event_only.discard(request_id)
        return self._request_queues.pop(request_id, None)

    def _fail(self, request_id: str, error: Exception) -> None:
        """Hand `error` to one waiting caller and forget the request."""
        queue = self._forget(request_id)
        if queue is not None:
            queue.put_nowait(error)

    def _fail_all(self, error: Exception) -> None:
        for request_id in list(self._request_queues):
            self._fail(request_id, error)

    def _abort(self, seqs: list[Sequence], error: Exception) -> None:
        for seq in seqs:
            seq.status = SequenceStatus.FINISHED_ABORTED
            self._fail(seq.request_id, error)

    def _forward(self, phase: Phase, seqs: list[Sequence], num_tokens: list[int]) -> list[Sequence]:
        """Run one forward pass over `num_tokens` per sequence, record it, and sample.

        Returns the sequences that sampled a token, which is all of them except a
        prompt chunk that stops short of its prompt's end: it has no next token
        yet, only K/V for the next chunk to attend to.

        Prefill and decode are separate passes, so a step that admits new
        sequences records two — which is what makes the two-pass cost (§1.3)
        visible in `stats`. `StepMetrics.wall_time_s` is the pass itself;
        sampling is charged to `stats.time.sample` instead, so the two are not
        conflated.
        """
        with self._timed("forward") as timer:
            if phase is Phase.PREFILL:
                logits = self.model_runner.prefill(seqs, num_tokens)
            else:
                logits = self.model_runner.decode(seqs)
        for seq, count in zip(seqs, num_tokens, strict=True):
            seq.num_computed_tokens += count

        with self._timed("sample"):
            rows = [i for i, seq in enumerate(seqs) if seq.is_prompt_computed]
            sampling_seqs = [seqs[i] for i in rows]
            if sampling_seqs:
                # Indexed only when a chunk is part-way, so the common step pays nothing.
                sampling_logits = logits if len(rows) == len(seqs) else logits[rows]
                sampled = self.sampler(sampling_logits, [seq.sampling_params for seq in sampling_seqs])
                self._apply_sampled(sampling_seqs, sampled)

        if self.config.collect_stats:
            self._record(phase, len(seqs), sum(num_tokens), len(sampling_seqs), timer.elapsed)
        return sampling_seqs

    def _record(
        self, phase: Phase, num_seqs: int, input_tokens: int, new_tokens: int, wall_time_s: float
    ) -> None:
        self.stats.record(
            StepMetrics(
                step_idx=self._step_idx,
                phase=phase,
                num_seqs=num_seqs,
                input_tokens=input_tokens,
                new_tokens=new_tokens,
                wall_time_s=wall_time_s,
                peak_gpu_mem_bytes=peak_gpu_memory_bytes(self.model_runner.device),
            )
        )
        self._step_idx += 1

    def _apply_sampled(self, seqs: list[Sequence], sampled) -> None:
        # One transfer for the batch: `.item()` per row is a separate
        # device-to-host copy, and each one stalls the pipeline behind it.
        for seq, token_id in zip(seqs, sampled.tolist(), strict=True):
            seq.output_token_ids.append(token_id)
            seq.detokenizer.update(self.tokenizer, seq.output_token_ids)
            self._maybe_finish(seq, token_id)

    def _build_event(self, seq: Sequence) -> StreamEvent:
        return StreamEvent(
            request_id=seq.request_id,
            prompt=seq.prompt,
            output_token_ids=list(seq.output_token_ids),
            text=seq.output_text,
            is_finished=seq.is_finished,
            finish_reason=_FINISH_REASONS.get(seq.status) if seq.is_finished else None,
        )

    def _maybe_finish(self, seq: Sequence, last_token_id: int) -> None:
        status = resolve_stop_status(seq, last_token_id, self.tokenizer, self.config.max_model_len)
        if status is not None:
            seq.status = status
