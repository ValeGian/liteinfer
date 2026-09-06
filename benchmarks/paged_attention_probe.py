"""Minimal Triton paged decode-attention, benchmarked against gather + fmha.

One program per (sequence, query head). Walks the sequence's slots in blocks,
reads K/V straight from the pool, and folds each block into a running softmax —
so the KV history is never copied into a contiguous tensor.
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode(Q, POOL_K, POOL_V, SLOTS, OUT,
                  stride_qb, stride_qh, stride_pn, stride_ph,
                  stride_sb, stride_ob, stride_oh,
                  KV_LEN, GROUPS, scale,
                  BLOCK: tl.constexpr, D: tl.constexpr):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUPS

    d = tl.arange(0, D)
    q = tl.load(Q + seq * stride_qb + head * stride_qh + d).to(tl.float32) * scale

    running_max = float("-inf")
    running_sum = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for start in range(0, KV_LEN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < KV_LEN
        slot = tl.load(SLOTS + seq * stride_sb + offs, mask=valid, other=0)
        base = slot[:, None] * stride_pn + kv_head * stride_ph + d[None, :]
        k = tl.load(POOL_K + base, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(POOL_V + base, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(valid, scores, float("-inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)

        acc = acc * rescale + tl.sum(weights[:, None] * v, axis=0)
        running_sum = running_sum * rescale + tl.sum(weights, axis=0)
        running_max = new_max

    tl.store(OUT + seq * stride_ob + head * stride_oh + d, (acc / running_sum).to(OUT.dtype.element_ty))


def paged_decode(q, pool_k, pool_v, slots, groups, scale, block=64):
    B, H, D = q.shape
    out = torch.empty_like(q)
    _paged_decode[(B, H)](
        q, pool_k, pool_v, slots, out,
        q.stride(0), q.stride(1), pool_k.stride(0), pool_k.stride(1),
        slots.stride(0), out.stride(0), out.stride(1),
        slots.shape[1], groups, scale, BLOCK=block, D=D,
    )
    return out


if __name__ == "__main__":
    from liteinfer.models.attention import _repeat_kv
    F = torch.nn.functional
    dev = torch.device("cuda")
    B, HQ, HKV, S, D = 32, 32, 8, 190, 64
    torch.manual_seed(0)
    pool_k = torch.randn(4096 * 16, HKV, D, dtype=torch.bfloat16, device=dev)
    pool_v = torch.randn(4096 * 16, HKV, D, dtype=torch.bfloat16, device=dev)
    slots = torch.randint(0, 4096 * 16, (B, S), device=dev)
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev)
    scale = D ** -0.5

    def reference():
        k = pool_k[slots].permute(0, 2, 1, 3)
        v = pool_v[slots].permute(0, 2, 1, 3)
        return F.scaled_dot_product_attention(
            q.unsqueeze(2), _repeat_kv(k, HQ // HKV), _repeat_kv(v, HQ // HKV), scale=scale
        ).squeeze(2)

    got, want = paged_decode(q, pool_k, pool_v, slots, HQ // HKV, scale), reference()
    print(f"max|diff| vs gather+fmha : {(got.float() - want.float()).abs().max().item():.3e}")

    def bench(fn, n=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / n * 1e3

    ref_ms = bench(reference)
    best = min((bench(lambda bs=bs: paged_decode(q, pool_k, pool_v, slots, HQ // HKV, scale, bs)), bs)
               for bs in (32, 64, 128))
    print(f"gather + fmha            : {ref_ms:7.4f} ms per layer")
    print(f"triton paged (BLOCK={best[1]:3d})  : {best[0]:7.4f} ms per layer   {ref_ms/best[0]:5.2f}x")
    print(f"per decode step (x16)    : {ref_ms*16:6.3f} -> {best[0]*16:6.3f} ms")
