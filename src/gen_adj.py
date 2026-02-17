#!/usr/bin/env python3
"""
gen_masks.py

Construct a global binary adjacency matrix A \in {0,1}^{N x N} for sparse attention.
A_ij = 1 iff token i attends to token j. (Paper: Global Adjacency Matrix Construction)

This script supports:
  1) Synthetic structured masks (window + optional global tokens)
  2) Loading masks from .npz/.npy/.pt (exported from real models)
  3) Optional aggregation across heads/layers (logical OR)

Outputs:
  - adjacency in SciPy CSR saved as .npz (recommended)
  - metadata JSON (N, pattern params)

Example:
  python -m src.gen_masks --mode window_global --N 4096 --window 64 --global_tokens 16 \
      --out results/graphs/A.npz
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception as e:
    raise RuntimeError("scipy is required for sparse adjacency. pip install scipy") from e

try:
    import torch
except Exception:
    torch = None


@dataclass
class MaskMeta:
    mode: str
    N: int
    window: Optional[int] = None
    global_tokens: Optional[int] = None
    seed: int = 0
    note: str = ""


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def build_window_global_mask(N: int, window: int, global_tokens: int) -> sp.csr_matrix:
    """
    Structured pattern: each token attends to a local window around itself (+ optional global tokens).

    - Local: token i attends to [i-window//2, i+window//2] clipped.
    - Global: every token attends to the first `global_tokens` tokens, and those tokens attend to all tokens.

    Output is CSR adjacency (binary).
    """
    assert window > 0
    assert global_tokens >= 0

    rows = []
    cols = []

    half = window // 2
    for i in range(N):
        j0 = max(0, i - half)
        j1 = min(N, i + half + 1)
        js = np.arange(j0, j1, dtype=np.int32)
        rows.append(np.full(js.shape, i, dtype=np.int32))
        cols.append(js)

        if global_tokens > 0:
            g = np.arange(0, min(global_tokens, N), dtype=np.int32)
            rows.append(np.full(g.shape, i, dtype=np.int32))
            cols.append(g)

    if global_tokens > 0:
        g = np.arange(0, min(global_tokens, N), dtype=np.int32)
        # global tokens attend to all tokens
        for gi in g:
            js = np.arange(0, N, dtype=np.int32)
            rows.append(np.full(js.shape, gi, dtype=np.int32))
            cols.append(js)

    row = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    col = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
    data = np.ones_like(row, dtype=np.uint8)

    A = sp.coo_matrix((data, (row, col)), shape=(N, N), dtype=np.uint8).tocsr()
    A.sum_duplicates()
    A.data[:] = 1
    return A


def load_mask_any(path: str) -> sp.csr_matrix:
    """
    Load adjacency/mask from:
      - .npz: scipy sparse saved via sp.save_npz
      - .npy: dense boolean/int matrix
      - .pt : torch tensor (dense or sparse COO)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        A = sp.load_npz(path).tocsr()
        A.data[:] = 1
        return A

    if ext == ".npy":
        M = np.load(path)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Expected square 2D array, got {M.shape}")
        A = sp.csr_matrix(M.astype(np.uint8))
        A.data[:] = 1
        return A

    if ext == ".pt":
        if torch is None:
            raise RuntimeError("torch is required to load .pt files. pip install torch")
        T = torch.load(path, map_location="cpu")
        if hasattr(T, "is_sparse") and T.is_sparse:
            T = T.coalesce()
            idx = T.indices().cpu().numpy()
            N = int(T.size(0))
            data = np.ones(idx.shape[1], dtype=np.uint8)
            A = sp.coo_matrix((data, (idx[0], idx[1])), shape=(N, N), dtype=np.uint8).tocsr()
            A.sum_duplicates()
            A.data[:] = 1
            return A
        else:
            M = np.array(T, dtype=np.uint8)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"Expected square 2D tensor, got {M.shape}")
            A = sp.csr_matrix(M)
            A.data[:] = 1
            return A

    raise ValueError(f"Unsupported mask format: {ext} (use .npz/.npy/.pt)")


def save_outputs(A: sp.csr_matrix, meta: MaskMeta, out_npz: str, out_meta_json: Optional[str]) -> None:
    _ensure_dir(out_npz)
    sp.save_npz(out_npz, A)

    if out_meta_json:
        _ensure_dir(out_meta_json)
        with open(out_meta_json, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="window_global",
                   choices=["window_global", "load"],
                   help="Mask construction mode.")
    p.add_argument("--N", type=int, default=4096, help="Number of tokens N.")
    p.add_argument("--window", type=int, default=64, help="Local window size (mode=window_global).")
    p.add_argument("--global_tokens", type=int, default=16, help="Number of global tokens (mode=window_global).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load_path", type=str, default="", help="Path to .npz/.npy/.pt (mode=load).")
    p.add_argument("--out", type=str, required=True, help="Output adjacency .npz path.")
    p.add_argument("--out_meta", type=str, default="", help="Optional metadata json path.")
    p.add_argument("--note", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if args.mode == "window_global":
        A = build_window_global_mask(args.N, args.window, args.global_tokens)
        meta = MaskMeta(mode=args.mode, N=args.N, window=args.window, global_tokens=args.global_tokens,
                        seed=args.seed, note=args.note)
    else:
        if not args.load_path:
            raise ValueError("--load_path is required when --mode load")
        A = load_mask_any(args.load_path)
        meta = MaskMeta(mode=args.mode, N=A.shape[0], seed=args.seed, note=f"loaded_from={args.load_path}; {args.note}")

    out_meta = args.out_meta if args.out_meta else None
    save_outputs(A, meta, args.out, out_meta)
    print(f"[OK] Saved adjacency: {args.out} (N={A.shape[0]}, nnz={A.nnz})")
    if out_meta:
        print(f"[OK] Saved meta: {out_meta}")


if __name__ == "__main__":
    main()
