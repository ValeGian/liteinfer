"""Stop-condition logic, shared by the sync and async engines."""

from __future__ import annotations

from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.tokenizer import Tokenizer


def resolve_stop_status(
    seq: Sequence,
    last_token_id: int,
    tokenizer: Tokenizer,
    max_model_len: int,
) -> SequenceStatus | None:
    """Return the terminal status for ``seq``, or ``None`` to keep generating.

    ``min_tokens`` gates every early-stop signal (EOS, stop token ids, stop
    strings) but never the length limits, so a sequence can always terminate.

    Stop strings are matched against ``seq.output_text``, which the caller has
    already advanced for this token — the text is decoded once per step and
    read here and by the stream event, rather than rebuilt by each.
    """
    params = seq.sampling_params
    num_output = seq.num_output_tokens

    if num_output >= params.min_tokens:
        hit_eos = last_token_id in tokenizer.eos_token_ids and not params.ignore_eos
        hit_stop_token = bool(params.stop_token_ids) and last_token_id in params.stop_token_ids
        if hit_eos or hit_stop_token:
            return SequenceStatus.FINISHED_STOPPED
        if params.stop and any(s in seq.output_text for s in params.stop):
            return SequenceStatus.FINISHED_STOPPED

    if num_output >= params.max_tokens or len(seq) >= max_model_len:
        return SequenceStatus.FINISHED_LENGTH
    return None
