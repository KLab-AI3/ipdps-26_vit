#!/usr/bin/env python3
"""
run_bench.py

Benchmark sparse attention execution at the extracted block granularity.

Paper intent:
- Blocks are the atomic units for scheduling and profiling (Lb,s etc.).
- Each block is executed with a single tile config during a launch (later you can
  swap in FlashAttention/xFormers kernels for real tile control).

This script:
1) loads blocks.json produced by extract_blocks.py
2) generates synthetic Q/K/V tensors (or loads from .pt)
3) runs per-block attention (dense or masked) and measures latency on GPU
4) writes results CSV/JSON

Notes:
- For correctness with "tolerate small sparsity within each block", you can:
    --assume_dense_blocks : treat each extracted block as dense (faster)
    default              : apply the true adjacency submask within the block (more correct)
- You can later integrate FlashAttention/xFormers by replacing the `attn_block()` function.

Example:
  python -m src.run_bench --adj results/graphs/A.npz --blocks results/blocks/blocks.json \
      --device cuda --dtype fp16 --head_dim 64 --heads 8 --iters 100 --warmup 10 \
      --out results/logs/bench.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("torch is required. pip install torch") from e

try:
    import scipy.sparse as sp
except Exception as e:
    raise RuntimeError("scipy is required. pip install scipy") from e


@dataclass
class Block:
    r0: int
    r1: int
    c0: int
    c1: int
    nnz: int
    area: int
    density: float


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def load_blocks_json(path: str) -> Tuple[int, List[Block]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    N = int(payload["N"])
    blocks = []
    for b in payload["blocks"]:
        blocks.append(Block(**b))
    return N, blocks


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_qkv(N: int, heads: int, head_dim: int, device: str, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Shape: [N, H, D]
    q = torch.randn((N, heads, head_dim), generator=g, device=device, dtype=dtype)
    k = torch.randn((N, heads, head_dim), generator=g, device=device, dtype=dtype)
    v = torch.randn((N, heads, head_dim), generator=g, device=device, dtype=dtype)
    return q, k, v


def attn_block(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute attention for one block.

    q: [BQ, H, D]
    k,v: [BK, H, D]
    mask: [BQ, BK] with True meaning "allowed", or None.

    Returns: out [BQ, H, D]
    """
    # torch SDPA expects [B, H, L, D] or [B, L, H, D] depending; we use [B,H,L,D]
    # Here we do a manual reshape:
    q4 = q.permute(1, 0, 2).unsqueeze(0)  # [1,H,BQ,D]
    k4 = k.permute(1, 0, 2).unsqueeze(0)  # [1,H,BK,D]
    v4 = v.permute(1, 0, 2).unsqueeze(0)  # [1,H,BK,D]

    if mask is not None:
        # SDPA uses additive mask: float(-inf) for disallowed.
        # mask True=allowed -> additive 0; False -> -inf
        attn_bias = torch.zeros((1, 1, mask.shape[0], mask.shape[1]), device=q.device, dtype=torch.float32)
        attn_bias.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    else:
        attn_bias = None

    # scaled_dot_product_attention is available in PyTorch 2.x
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False
    )  # [1,H,BQ,D]

    return out.squeeze(0).permute(1, 0, 2).contiguous()  # [BQ,H,D]


def build_block_mask_from_adj(A: sp.csr_matrix, blk: Block, device: str) -> torch.Tensor:
    """
    Extract the exact adjacency submask within the block:
      M[u,v] = True iff A[r0+u, c0+v] == 1
    """
    sub = A[blk.r0:blk.r1, blk.c0:blk.c1].tocsr()
    # Build dense boolean mask (BQ x BK) for correctness.
    # If blocks are huge, you may want a sparse bias instead (xFormers).
    M = sub.toarray().astype(np.bool_)
    return torch.from_numpy(M).to(device=device)


def time_cuda(fn, warmup: int, iters: int) -> Tuple[float, float]:
    """
    Returns (mean_ms, p50_ms) over iters (after warmup).
    """
    if not torch.cuda.is_available():
        # CPU timing fallback
        times = []
        for _ in range(warmup):
            fn()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times = np.array(times)
        return float(times.mean()), float(np.median(times))

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    times = np.array(times, dtype=np.float64)
    return float(times.mean()), float(np.median(times))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adj", type=str, required=True, help="Adjacency .npz used to build per-block masks.")
    p.add_argument("--blocks", type=str, required=True, help="blocks.json from extract_blocks.py")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--dtype", type=str, default="fp16", help="fp16|bf16|fp32")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--assume_dense_blocks", action="store_true",
                   help="If set, skip internal block masks and treat each block as dense.")
    p.add_argument("--max_blocks", type=int, default=0, help="If >0, benchmark only first K blocks.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    p.add_argument("--out_json", type=str, default="", help="Optional output JSON (full metadata).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = get_dtype(args.dtype)

    A = sp.load_npz(args.adj).tocsr()
    N, blocks = load_blocks_json(args.blocks)
    if A.shape[0] != N:
        raise ValueError(f"Adjacency N={A.shape[0]} != blocks.json N={N}")

    if args.max_blocks and args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]

    q, k, v = make_qkv(N, args.heads, args.head_dim, device=args.device, dtype=dtype, seed=args.seed)

    _ensure_dir(args.out)
    rows: List[Dict[str, Any]] = []

    # Optional: clear memory stats
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for bi, blk in enumerate(blocks):
        BQ = blk.r1 - blk.r0
        BK = blk.c1 - blk.c0

        qb = q[blk.r0:blk.r1]
        kb = k[blk.c0:blk.c1]
        vb = v[blk.c0:blk.c1]

        mask_t = None
        if (not args.assume_dense_blocks):
            mask_t = build_block_mask_from_adj(A, blk, device=args.device)

        def _run_once():
            out = attn_block(qb, kb, vb, mask=mask_t)
            # prevent compiler from dropping work
            return out

        mean_ms, p50_ms = time_cuda(_run_once, warmup=args.warmup, iters=args.iters)

        peak_mem = None
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            peak_mem = int(torch.cuda.max_memory_allocated())

        rows.append({
            "block_id": bi,
            "r0": blk.r0, "r1": blk.r1, "c0": blk.c0, "c1": blk.c1,
            "BQ": BQ, "BK": BK,
            "nnz": blk.nnz, "area": blk.area, "density": blk.density,
            "dtype": args.dtype, "heads": args.heads, "head_dim": args.head_dim,
            "assume_dense_blocks": int(args.assume_dense_blocks),
            "mean_ms": mean_ms,
            "p50_ms": p50_ms,
            "peak_mem_bytes": peak_mem if peak_mem is not None else "",
        })

        if (bi + 1) % 50 == 0:
            print(f"[{bi+1}/{len(blocks)}] mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  BQxBK={BQ}x{BK}")

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote: {args.out} (blocks={len(rows)})")

    if args.out_json:
        _ensure_dir(args.out_json)
        payload = {
            "adj": args.adj,
            "blocks": args.blocks,
            "device": args.device,
            "dtype": args.dtype,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "warmup": args.warmup,
            "iters": args.iters,
            "assume_dense_blocks": args.assume_dense_blocks,
            "num_blocks": len(rows),
            "rows": rows,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
