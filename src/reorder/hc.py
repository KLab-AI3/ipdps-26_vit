"""
Hierarchical Clustering (HC) reordering for sparse attention adjacency matrices.

- Input: A (scipy.sparse CSR), binary adjacency matrix.
- Output: perm (List[int]) permutation of [0..N-1], intended for symmetric reorder.

HC description used here:
- Start from singleton clusters, each cluster has a "pattern" = union of member rows.
- Repeatedly merge the closest pair of clusters by a chosen distance metric
  (Hamming/Jaccard on quotient patterns), until a stopping condition is met.

This follows the bottom-up greedy HC described in:
- Your paper (HC/IC/RCM section): merge closest clusters until stopping condition. 
- "Blocking Sparse Matrices ..." idea: represent cluster by union pattern.

Stopping condition:
- default: stop when average cluster size reaches target_avg_cluster_size
  OR when number of clusters <= min_num_clusters.

Note: Full exact HC can be expensive (O(N^2)). We restrict candidate pairs to
clusters that share at least one quotient block, which is well-aligned with the goal
of grouping similar rows and is practical for N up to ~8192.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import heapq
import numpy as np

from scipy import sparse



def _iter_set_bits(x: int) -> Iterable[int]:
    while x:
        lsb = x & -x
        yield (lsb.bit_length() - 1)
        x ^= lsb


def build_quotient_bitrows(A: "sparse.csr_matrix", col_block_w: int) -> List[int]:
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    n_rows, n_cols = A.shape
    bitrows: List[int] = [0] * n_rows
    indptr = A.indptr
    indices = A.indices
    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        q = 0
        for c in indices[start:end]:
            q |= (1 << (c // col_block_w))
        bitrows[i] = q
    return bitrows


def jaccard_distance(a: int, b: int) -> float:
    inter = (a & b).bit_count()
    union = (a | b).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def hamming_distance_sets(a: int, b: int) -> int:
    """
    Hamming-like distance on set representation (bitset):
    |a \\ b| + |b \\ a|  == popcount(a xor b).
    """
    return (a ^ b).bit_count()


@dataclass
class HCConfig:
    col_block_w: int = 16
    metric: str = "hamming"  # "hamming" or "jaccard"
    target_avg_cluster_size: int = 8
    min_num_clusters: int = 1
    # candidate restriction: only consider pairs that share blocks
    min_shared_blocks: int = 1
    # optional cap on considered neighbors per cluster to keep heap small
    max_neighbors_per_cluster: int = 64
    random_seed: int = 0


class _Cluster:
    __slots__ = ("cid", "members", "pattern", "alive")

    def __init__(self, cid: int, members: List[int], pattern: int):
        self.cid = cid
        self.members = members
        self.pattern = pattern
        self.alive = True


def hierarchical_clustering_order(A: "sparse.csr_matrix", cfg: HCConfig) -> List[int]:
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Attention adjacency should be square (N x N)."

    bitrows = build_quotient_bitrows(A, cfg.col_block_w)

    # build initial clusters
    clusters: Dict[int, _Cluster] = {i: _Cluster(i, [i], bitrows[i]) for i in range(n)}
    alive: Set[int] = set(clusters.keys())

    # postings: block -> clusters currently touching block
    postings: Dict[int, Set[int]] = {}
    for cid, c in clusters.items():
        for b in _iter_set_bits(c.pattern):
            postings.setdefault(b, set()).add(cid)

    def dist(p1: int, p2: int) -> float:
        if cfg.metric == "jaccard":
            return jaccard_distance(p1, p2)
        if cfg.metric == "hamming":
            return float(hamming_distance_sets(p1, p2))
        raise ValueError(f"Unknown metric: {cfg.metric}")

    # heap items: (distance, cid1, cid2, version_tag)
    heap: List[Tuple[float, int, int]] = []

    def candidate_neighbors(cid: int) -> List[int]:
        """Return candidate cluster IDs that share >= min_shared_blocks with cid."""
        c = clusters[cid]
        counts: Dict[int, int] = {}
        for b in _iter_set_bits(c.pattern):
            for other in postings.get(b, ()):
                if other == cid or other not in alive:
                    continue
                counts[other] = counts.get(other, 0) + 1

        # filter by shared-blocks
        cands = [k for k, v in counts.items() if v >= cfg.min_shared_blocks]
        # keep a bounded number of neighbors (prefer more shared blocks)
        cands.sort(key=lambda x: counts[x], reverse=True)
        return cands[: cfg.max_neighbors_per_cluster]

    # initialize heap with nearest-ish neighbors for each cluster
    for cid in list(alive):
        for nb in candidate_neighbors(cid):
            if cid < nb:
                heapq.heappush(heap, (dist(clusters[cid].pattern, clusters[nb].pattern), cid, nb))

    def avg_cluster_size() -> float:
        if not alive:
            return float(n)
        return n / float(len(alive))

    next_cid = n

    # main merge loop
    while len(alive) > cfg.min_num_clusters and avg_cluster_size() < cfg.target_avg_cluster_size:
        if not heap:
            break

        d, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive:
            continue

        ca, cb = clusters[a], clusters[b]
        # recompute distance to be safe (patterns may have changed if we had stale edges)
        d2 = dist(ca.pattern, cb.pattern)
        if d2 != d:
            heapq.heappush(heap, (d2, a, b))
            continue

        # merge
        new_members = ca.members + cb.members
        new_pattern = ca.pattern | cb.pattern
        new_id = next_cid
        next_cid += 1

        clusters[new_id] = _Cluster(new_id, new_members, new_pattern)

        # deactivate old clusters
        alive.remove(a)
        alive.remove(b)
        ca.alive = False
        cb.alive = False

        # remove old postings and add new postings
        for bb in _iter_set_bits(ca.pattern):
            if bb in postings:
                postings[bb].discard(a)
        for bb in _iter_set_bits(cb.pattern):
            if bb in postings:
                postings[bb].discard(b)

        alive.add(new_id)
        for bb in _iter_set_bits(new_pattern):
            postings.setdefault(bb, set()).add(new_id)

        # add heap edges for the new cluster
        for nb in candidate_neighbors(new_id):
            lo, hi = (new_id, nb) if new_id < nb else (nb, new_id)
            heapq.heappush(heap, (dist(clusters[lo].pattern, clusters[hi].pattern), lo, hi))

    # produce a permutation:
    # clusters are unordered; we sort clusters by the smallest original index in each cluster
    final_clusters = [clusters[cid] for cid in alive]
    final_clusters.sort(key=lambda c: min(c.members))

    perm: List[int] = []
    for c in final_clusters:
        perm.extend(c.members)

    assert len(perm) == n
    return perm


def apply_symmetric_permutation(A: "sparse.csr_matrix", perm: List[int]) -> "sparse.csr_matrix":
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    p = np.asarray(perm, dtype=np.int64)
    return A[p][:, p].tocsr()

