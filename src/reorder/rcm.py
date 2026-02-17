"""
RCM reordering for sparse attention adjacency graphs (paper-ready).

Input:
  - edges: list of (i,j) pairs indicating nonzero attention connections
    OR
  - dense_mask: (N,N) array-like where nonzero indicates connection

Process:
  1) Build sparse adjacency A (CSR)
  2) RCM permutation using SciPy's reverse_cuthill_mckee
     - symmetric_mode=True (recommended for attention graphs)
  3) Reorder adjacency: A_rcm = A[perm][:, perm]
  4) Return perm/invperm + optional bandwidth stats

Outputs:
  - perm: np.ndarray, shape (N,), mapping new_position -> old_index
  - invperm: np.ndarray, shape (N,), mapping old_index -> new_position
  - A_rcm: scipy.sparse.csr_matrix (optional)
  - bandwidth_before/after: ints (optional)

Dependencies:
  pip install scipy numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


@dataclass
class RCMResult:
    perm: np.ndarray        # new_pos -> old_idx
    invperm: np.ndarray     # old_idx -> new_pos
    bandwidth_before: int
    bandwidth_after: int
    A_rcm: Optional[csr_matrix] = None


def _bandwidth_csr(A: csr_matrix) -> int:
    """
    Bandwidth = max_{(i,j) in nnz} |i-j|
    (works for directed or undirected; uses stored nnz)
    """
    A = A.tocsr()
    bw = 0
    indptr = A.indptr
    indices = A.indices
    for i in range(A.shape[0]):
        row_js = indices[indptr[i]:indptr[i+1]]
        if row_js.size == 0:
            continue
        # max abs diff in that row
        diff = np.max(np.abs(row_js - i))
        if diff > bw:
            bw = int(diff)
    return bw


def build_adjacency_from_edges(
    n: int,
    edges: Iterable[Tuple[int, int]],
    make_binary: bool = True,
) -> csr_matrix:
    """
    Build sparse adjacency CSR from (i,j) edges.
    - For attention masks, edges can be directed (i->j).
    - Values are 1 (binary) by default.
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for (i, j) in edges:
        if i < 0 or j < 0 or i >= n or j >= n or i == j:
            continue
        rows.append(int(i))
        cols.append(int(j))
        vals.append(1.0 if make_binary else 1.0)

    if not rows:
        return csr_matrix((n, n), dtype=np.float32)

    A = coo_matrix((np.array(vals, dtype=np.float32),
                    (np.array(rows, dtype=np.int32),
                     np.array(cols, dtype=np.int32))),
                   shape=(n, n)).tocsr()
    # remove duplicates by summing -> then binarize if needed
    A.sum_duplicates()
    if make_binary:
        A.data[:] = 1.0
    return A


def build_adjacency_from_dense_mask(dense_mask: Union[np.ndarray, Sequence[Sequence[int]]]) -> csr_matrix:
    """
    Build CSR adjacency from a dense (N,N) 0/1 mask.
    Nonzero => edge.
    """
    M = np.asarray(dense_mask)
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "dense_mask must be square (N,N)."
    # treat nonzero as 1
    M = (M != 0).astype(np.float32)
    return csr_matrix(M)


def rcm_reorder_attention_graph(
    n: int,
    *,
    edges: Optional[Iterable[Tuple[int, int]]] = None,
    dense_mask: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    symmetric_mode: bool = True,
    return_reordered_adjacency: bool = True,
) -> RCMResult:
    """
    Compute RCM ordering suitable for attention adjacency graphs.

    symmetric_mode=True is recommended:
      - Attention edges are typically directed (Q->K).
      - RCM is defined for undirected graphs.
      - symmetric_mode uses (A + A^T) for ordering internally.

    Note:
      perm returned is the SciPy convention: array of node indices in new order
      (new_pos -> old_idx). To map old -> new, use invperm.
    """
    if (edges is None) == (dense_mask is None):
        raise ValueError("Provide exactly one of edges or dense_mask.")

    if edges is not None:
        A = build_adjacency_from_edges(n, edges, make_binary=True)
    else:
        A = build_adjacency_from_dense_mask(dense_mask)
        n = A.shape[0]

    bw_before = _bandwidth_csr(A)

    # SciPy RCM
    perm = reverse_cuthill_mckee(A, symmetric_mode=symmetric_mode).astype(np.int32)
    invperm = np.empty_like(perm)
    invperm[perm] = np.arange(n, dtype=np.int32)

    A_rcm = None
    bw_after = bw_before
    if return_reordered_adjacency:
        A_rcm = A[perm, :][:, perm].tocsr()
        bw_after = _bandwidth_csr(A_rcm)

    return RCMResult(
        perm=perm,
        invperm=invperm,
        bandwidth_before=bw_before,
        bandwidth_after=bw_after,
        A_rcm=A_rcm,
    )


def apply_perm_to_edges(
    edges: Iterable[Tuple[int, int]],
    invperm: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Convert original edges (old indices) into reordered-index edges.
    After reordering, node old_i moves to new index invperm[old_i].
    """
    out: List[Tuple[int, int]] = []
    for i, j in edges:
        ni = int(invperm[i])
        nj = int(invperm[j])
        if ni != nj:
            out.append((ni, nj))
    return out


# -----------------------------
# Minimal CLI for your pipeline
# -----------------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of tokens/nodes")
    ap.add_argument("--edges_json", type=str, default="", help="path to edges json: [[i,j], ...]")
    ap.add_argument("--out_prefix", type=str, default="results/graphs/rcm", help="output prefix")
    ap.add_argument("--no_adj", action="store_true", help="do not output reordered adjacency npz")
    args = ap.parse_args()

    if not args.edges_json:
        raise SystemExit("Provide --edges_json with [[i,j], ...].")

    with open(args.edges_json, "r") as f:
        edges = json.load(f)
    edges = [(int(i), int(j)) for i, j in edges]

    res = rcm_reorder_attention_graph(
        args.n,
        edges=edges,
        symmetric_mode=True,
        return_reordered_adjacency=(not args.no_adj),
    )

    print(f"[RCM] bandwidth: {res.bandwidth_before} -> {res.bandwidth_after}")
    np.save(args.out_prefix + "_perm.npy", res.perm)
    np.save(args.out_prefix + "_invperm.npy", res.invperm)

    # Optionally store adjacency
    if res.A_rcm is not None:
        # save sparse matrix
        from scipy.sparse import save_npz
        save_npz(args.out_prefix + "_adj.npz", res.A_rcm)

    print(f"[OK] wrote {args.out_prefix}_perm.npy, {args.out_prefix}_invperm.npy")
