"""
Iterative Clustering (IC) reordering for sparse attention adjacency matrices.

- Input: A (scipy.sparse CSR), binary adjacency matrix A in {0,1}^{N x N}
  where A[i, j] = 1 indicates token i attends to token j.
- Output: perm (List[int]) permutation of [0..N-1].
  For attention adjacency, we usually apply symmetric permutation:
      A_reordered = A[perm][:, perm]

This implements the seed-based Iterative Clustering described in:
- Your paper: IC selects a seed row, aggregates other rows within threshold tau,
  repeats until all rows assigned. (HC/IC/RCM section)
- "Blocking Sparse Matrices ..." IC uses distance to cluster pattern (union of rows).

Distance:
- Default is Jaccard distance between a row and cluster pattern:
    J(v, p) = 1 - |v∩p|/|v∪p|
You can switch to Hamming-like distance if desired.

Scalability notes:
- Pure IC is O(NK) worst-case (K nnz). We accelerate candidate discovery using
  an inverted index over quotient-block IDs (see build_postings()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from scipy import sparse



def _iter_set_bits(x: int) -> Iterable[int]:
    """Yield indices of set bits in an integer bitset."""
    while x:
        lsb = x & -x
        yield (lsb.bit_length() - 1)
        x ^= lsb


def build_quotient_bitrows(A: "sparse.csr_matrix", col_block_w: int) -> List[int]:
    """
    Convert CSR adjacency matrix into quotient-bitrows (bitset per row).
    Bit j means: row has >=1 nnz in columns [j*W, (j+1)*W).

    This mirrors the 'quotient row' idea with fixed column partition width W.
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    n_rows, n_cols = A.shape
    n_qcols = (n_cols + col_block_w - 1) // col_block_w

    bitrows: List[int] = [0] * n_rows
    indptr = A.indptr
    indices = A.indices

    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        row_cols = indices[start:end]
        q = 0
        # mark quotient blocks touched by any nnz
        for c in row_cols:
            qb = c // col_block_w
            q |= (1 << qb)
        bitrows[i] = q
    # sanity: if n_qcols > Python int bit-length, still fine (Python int arbitrary precision)
    _ = n_qcols
    return bitrows


def build_postings(bitrows: List[int]) -> Dict[int, List[int]]:
    """
    Inverted index: quotient-block id -> list of rows that touch that block.
    Helps IC quickly find candidates that share at least one block with current pattern.
    """
    postings: Dict[int, List[int]] = {}
    for r, bm in enumerate(bitrows):
        for b in _iter_set_bits(bm):
            postings.setdefault(b, []).append(r)
    return postings


def jaccard_distance_bitset(a: int, b: int) -> float:
    """Jaccard distance between two bitsets a and b."""
    inter = (a & b).bit_count()
    union = (a | b).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


@dataclass
class ICConfig:
    col_block_w: int = 16         # quotient block width W
    tau: float = 0.6              # distance threshold (<= tau => merge)
    max_cluster_size: Optional[int] = None  # optional cap (helps avoid huge clusters)
    candidate_expand_rounds: int = 2        # how many times to expand candidate set
    min_shared_blocks: int = 1     # candidates must share >= this many quotient blocks
    seed_order: str = "natural"    # "natural" or "random"
    random_seed: int = 0


def iterative_clustering_order(
    A: "sparse.csr_matrix",
    cfg: ICConfig,
) -> List[int]:
    """
    Returns a permutation produced by Iterative Clustering.

    High-level:
      V = all rows
      while V not empty:
         pick seed v
         c = {v}, pattern p = v
         for each candidate w in V:
             if dist(w, p) <= tau:
                 add w to cluster, update p = p union w
    We accelerate by only scanning candidates that share blocks with current pattern.
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Attention adjacency should be square (N x N)."

    bitrows = build_quotient_bitrows(A, cfg.col_block_w)
    postings = build_postings(bitrows)

    unassigned: Set[int] = set(range(n))
    perm: List[int] = []

    rng = np.random.default_rng(cfg.random_seed)
    seeds = list(range(n))
    if cfg.seed_order == "random":
        rng.shuffle(seeds)

    for seed in seeds:
        if seed not in unassigned:
            continue

        # start new cluster
        cluster: List[int] = [seed]
        unassigned.remove(seed)
        pattern = bitrows[seed]

        # candidate discovery via postings
        def gather_candidates(cur_pattern: int) -> Set[int]:
            cand: Set[int] = set()
            # union posting lists for blocks present in pattern
            for b in _iter_set_bits(cur_pattern):
                for r in postings.get(b, []):
                    if r in unassigned:
                        cand.add(r)
            return cand

        candidates = gather_candidates(pattern)

        # optionally expand candidates as pattern grows
        for _round in range(cfg.candidate_expand_rounds):
            if not candidates:
                break

            # scan candidates (stable order for reproducibility)
            cand_list = sorted(candidates)
            candidates.clear()

            for w in cand_list:
                if w not in unassigned:
                    continue

                # quick filter: require some shared blocks
                shared = (bitrows[w] & pattern).bit_count()
                if shared < cfg.min_shared_blocks:
                    continue

                dist = jaccard_distance_bitset(bitrows[w], pattern)
                if dist <= cfg.tau:
                    cluster.append(w)
                    unassigned.remove(w)
                    pattern |= bitrows[w]

                    if cfg.max_cluster_size is not None and len(cluster) >= cfg.max_cluster_size:
                        break

            if cfg.max_cluster_size is not None and len(cluster) >= cfg.max_cluster_size:
                break

            # re-gather with expanded pattern
            candidates = gather_candidates(pattern)

        # append cluster to permutation
        perm.extend(cluster)


def apply_symmetric_permutation(A: "sparse.csr_matrix", perm: List[int]) -> "sparse.csr_matrix":
    """Return A[perm][:, perm] as CSR."""
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    p = np.asarray(perm, dtype=np.int64)
    return A[p][:, p].tocsr()

