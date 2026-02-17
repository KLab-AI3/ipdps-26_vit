"""
extract_blocks.py

Density-aware block extraction from a (reordered) adjacency matrix A.

Paper behavior:
- Start from each unvisited nonzero element.
- Expand rightward and downward while keeping density rho >= rho_min,
  where rho = nnz / (h*w).
- Finalize a block when rho would fall below rho_min; repeat until all nnz covered.
- Tolerate some sparsity within each block to improve GPU friendliness.

Inputs:
  - adjacency matrix in SciPy CSR (.npz)
Outputs:
  - blocks.json: list of blocks with (r0,r1,c0,c1, nnz, area, density)

Example:
  python -m src.extract_blocks --adj results/graphs/A_reordered.npz --rho_min 0.6 \
      --max_h 1024 --max_w 1024 --out results/blocks/blocks.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

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
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def nnz_in_rect_csr(A: sp.csr_matrix, r0: int, r1: int, c0: int, c1: int) -> int:
    """
    Count nnz in [r0,r1) x [c0,c1) using CSR row slices + binary search.
    Complexity: O((r1-r0) * log nnz_row)
    """
    indptr = A.indptr
    indices = A.indices
    total = 0
    for r in range(r0, r1):
        start, end = indptr[r], indptr[r + 1]
        row_cols = indices[start:end]
        if row_cols.size == 0:
            continue
        left = np.searchsorted(row_cols, c0, side="left")
        right = np.searchsorted(row_cols, c1, side="left")
        total += int(right - left)
    return total


def mark_visited_in_rect(visited_flags: List[np.ndarray], A: sp.csr_matrix, r0: int, r1: int, c0: int, c1: int) -> None:
    """
    Mark CSR positions as visited for nnz entries that fall within the rectangle.
    visited_flags[r] is a boolean array of length nnz(row r), aligned with A.indices slice.
    """
    indptr = A.indptr
    indices = A.indices
    for r in range(r0, r1):
        start, end = indptr[r], indptr[r + 1]
        row_cols = indices[start:end]
        if row_cols.size == 0:
            continue
        left = np.searchsorted(row_cols, c0, side="left")
        right = np.searchsorted(row_cols, c1, side="left")
        if right > left:
            visited_flags[r][left:right] = True


def find_next_seed(A: sp.csr_matrix, visited_flags: List[np.ndarray], start_row: int = 0) -> Tuple[int, int]:
    """
    Return (r, c) of the next unvisited nnz, scanning rows from start_row.
    If none, return (-1, -1).
    """
    indptr = A.indptr
    indices = A.indices
    nrows = A.shape[0]
    for r in range(start_row, nrows):
        s, e = indptr[r], indptr[r + 1]
        if e <= s:
            continue
        flags = visited_flags[r]
        if flags.all():
            continue
        pos = int(np.argmax(~flags))  # first False
        c = int(indices[s + pos])
        return r, c
    return -1, -1


def greedy_expand_block(
    A: sp.csr_matrix,
    seed_r: int,
    seed_c: int,
    rho_min: float,
    max_h: int,
    max_w: int,
) -> Block:
    """
    Expand a rectangle starting at (seed_r, seed_c).
    Strategy:
      - Start with 1x1.
      - Alternate trying to grow width and height greedily:
        choose the expansion (right or down) that keeps rho>=rho_min and yields larger area;
        stop when neither expansion is possible.
    """
    r0, c0 = seed_r, seed_c
    r1, c1 = seed_r + 1, seed_c + 1

    nnz = nnz_in_rect_csr(A, r0, r1, c0, c1)
    area = (r1 - r0) * (c1 - c0)
    rho = nnz / max(area, 1)

    # If the seed isn't actually nnz (shouldn't happen), fallback.
    if nnz == 0:
        return Block(r0, r1, c0, c1, 0, area, 0.0)

    changed = True
    while changed:
        changed = False

        # Candidate: expand right by 1
        can_right = (c1 - c0) < max_w and c1 < A.shape[1]
        nnz_right = None
        rho_right = -1.0
        if can_right:
            nnz_right = nnz_in_rect_csr(A, r0, r1, c0, c1 + 1)
            area_right = (r1 - r0) * ((c1 + 1) - c0)
            rho_right = nnz_right / max(area_right, 1)

        # Candidate: expand down by 1
        can_down = (r1 - r0) < max_h and r1 < A.shape[0]
        nnz_down = None
        rho_down = -1.0
        if can_down:
            nnz_down = nnz_in_rect_csr(A, r0, r1 + 1, c0, c1)
            area_down = ((r1 + 1) - r0) * (c1 - c0)
            rho_down = nnz_down / max(area_down, 1)

        # Choose better expansion that satisfies rho_min
        best = None
        if can_right and rho_right >= rho_min:
            best = ("right", nnz_right, rho_right, (r1 - r0) * ((c1 + 1) - c0))
        if can_down and rho_down >= rho_min:
            cand = ("down", nnz_down, rho_down, ((r1 + 1) - r0) * (c1 - c0))
            if best is None or cand[3] > best[3]:
                best = cand

        if best is None:
            break

        direction, nnz_new, rho_new, _area_new = best
        if direction == "right":
            c1 += 1
        else:
            r1 += 1
        nnz = int(nnz_new)
        rho = float(rho_new)
        changed = True

    area = (r1 - r0) * (c1 - c0)
    return Block(r0, r1, c0, c1, nnz, area, rho)


def extract_blocks(A: sp.csr_matrix, rho_min: float, max_h: int, max_w: int) -> List[Block]:
    """
    Main extraction loop: repeatedly pick an unvisited nnz seed, grow a block, mark visited.
    """
    A = A.tocsr()
    A.sum_duplicates()
    A.data[:] = 1

    n = A.shape[0]
    visited_flags: List[np.ndarray] = []
    for r in range(n):
        row_nnz = A.indptr[r + 1] - A.indptr[r]
        visited_flags.append(np.zeros(row_nnz, dtype=bool))

    blocks: List[Block] = []
    scan_row = 0

    while True:
        sr, sc = find_next_seed(A, visited_flags, start_row=scan_row)
        if sr < 0:
            break

        blk = greedy_expand_block(A, sr, sc, rho_min=rho_min, max_h=max_h, max_w=max_w)

        # Mark all nnz in blk as visited.
        mark_visited_in_rect(visited_flags, A, blk.r0, blk.r1, blk.c0, blk.c1)
        blocks.append(blk)

        scan_row = blk.r0  # keep scanning near current region

    return blocks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adj", type=str, required=True, help="Input adjacency .npz (CSR).")
    p.add_argument("--rho_min", type=float, default=0.6, help="Minimum density threshold rho_min.")
    p.add_argument("--max_h", type=int, default=1024, help="Max block height.")
    p.add_argument("--max_w", type=int, default=1024, help="Max block width.")
    p.add_argument("--out", type=str, required=True, help="Output blocks.json path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    A = sp.load_npz(args.adj).tocsr()
    blocks = extract_blocks(A, rho_min=args.rho_min, max_h=args.max_h, max_w=args.max_w)

    _ensure_dir(args.out)
    payload = {
        "adj_path": args.adj,
        "N": int(A.shape[0]),
        "rho_min": float(args.rho_min),
        "max_h": int(args.max_h),
        "max_w": int(args.max_w),
        "num_blocks": len(blocks),
        "blocks": [asdict(b) for b in blocks],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    nnz_total = int(A.nnz)
    nnz_cov = sum(b.nnz for b in blocks)
    print(f"[OK] blocks={len(blocks)}  A.nnz={nnz_total}  sum(block.nnz)={nnz_cov}")
    print(f"[OK] saved: {args.out}")


if __name__ == "__main__":
    main()
