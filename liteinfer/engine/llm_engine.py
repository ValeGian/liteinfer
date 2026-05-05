"""LLMEngine — orchestrates scheduling, execution, and KV cache lifecycle."""

from __future__ import annotations

from itertools import count

from liteinfer.config import EngineConfig
from liteinfer.engine.metrics import (
    EngineStats,
    Phase,
    StepMetrics,
    StepTimer,
    peak_gpu_memory_bytes,
)
from liteinfer.engine.model_runner import ModelRunner
from liteinfer.engine.scheduler import Scheduler
from liteinfer.engine.sequence import (
    Sequence,
    SequenceGroup,
    SequenceStatus,
)
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler
from liteinfer.tokenizer import Tokenizer


class LLMEngine:
    """Core inference engine. One forward pass per `step()`."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config)
        self.sampler = Sampler()
        self.stats = EngineStats()
        self._seq_id_gen = count(0)
        self._step_idx = 0

    @property
    def tokenizer(self) -> Tokenizer:
        return self.model_runner.tokenizer

    def load_model(self) -> None:
        self.model_runner.load_model()

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Tokenize, wrap as a `SequenceGroup`, and enqueue with the scheduler."""
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens, >= max_model_len={self.config.max_model_len}")
        seq = Sequence(seq_id=next(self._seq_id_gen), prompt_token_ids=list(token_ids))
        group = SequenceGroup(
            request_id=request_id,
            sequences=[seq],
            sampling_params=sampling_params,
            prompt=prompt,
        )
        self.scheduler.add(group)

    def step(self) -> list[SequenceGroup]:
        """Run one schedule + forward iteration. Return finished groups (if any)."""
        sched_out = self.scheduler.schedule()
        if not sched_out.scheduled:
            return []

        if sched_out.is_new_batch:
            self.model_runner.start_batch(sched_out.scheduled)

        phase = self._phase_for(sched_out.is_new_batch)

        with StepTimer(self.model_runner.device) as timer:
            logits, input_tokens = self.model_runner.execute(sched_out.scheduled, is_new_batch=sched_out.is_new_batch)
            sampling_params = [g.sampling_params for g in sched_out.scheduled]
            sampled = self.sampler(logits, sampling_params)

        new_tokens = self._apply_sampled(sched_out.scheduled, sampled)

        if self.config.collect_stats:
            metrics = StepMetrics(
                step_idx=self._step_idx,
                phase=phase,
                num_seqs=len(sched_out.scheduled),
                input_tokens=input_tokens,
                new_tokens=new_tokens,
                wall_time_s=timer.elapsed,
                peak_gpu_mem_bytes=peak_gpu_memory_bytes(self.model_runner.device),
            )
            self.stats.record(metrics)
        self._step_idx += 1

        finished = self.scheduler.remove_finished()
        self.stats.num_requests_finished += len(finished)
        if not self.scheduler.running:
            self.model_runner.end_batch()
        return finished

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def _phase_for(self, is_new_batch: bool) -> Phase:
        if self.config.cache_mode == "none":
            return Phase.RECOMPUTE
        return Phase.PREFILL if is_new_batch else Phase.DECODE

    def _apply_sampled(
        self,
        scheduled: list[SequenceGroup],
        sampled,
    ) -> int:
        """Append sampled tokens, update statuses. Returns count of new tokens."""
        new_count = 0
        for i, group in enumerate(scheduled):
            seq = group.primary
            if seq.is_finished:
                continue
            token_id = int(sampled[i].item())
            seq.output_token_ids.append(token_id)
            new_count += 1
            self._maybe_finish(seq, group.sampling_params, token_id)
        return new_count

    def _maybe_finish(
        self,
        seq: Sequence,
        params: SamplingParams,
        last_token_id: int,
    ) -> None:
        if last_token_id in self.tokenizer.eos_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
        if params.stop_token_ids and last_token_id in params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return
        if seq.num_output_tokens >= params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH
            return
        if params.stop:
            text = self.tokenizer.decode(seq.output_token_ids)
            if any(s in text for s in params.stop):
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if len(seq) >= self.config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH
