"""
Solve the IPDPS sparse-attention scheduling ILP with Gurobi.

Decision variables:
  x[b,s,r] ∈ {0,1}  block b assigned to round r with tile s
  y[s,r]   ∈ {0,1}  tile s activated in round r
  Tr[r]    ≥ 0      duration of round r
  T        ≥ 0      makespan (max over Tr)

Constraints (C1-C7): assignment, feasibility, tile-cap (kappa),
round latency lower bound, round resource envelope, SM occupancy,
tile conflict matrix.

Input formats:
  - profiling_csv: rows = (block_id, tile_id, latency, smem, reg, warp, occ_sm, compat)
  - tiles_csv (optional): per-tile envelope: (tile_id, smem_env, reg_env, warp_env)
    If omitted, envelopes default to max over blocks for that tile in profiling_csv.
  - conflicts_csv (optional): (tile_i, tile_j, conflict_01) for i<j. Missing => 0.
Outputs:
  - schedule.json: block -> {round, tile}, plus round activations and makespan
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from typing import Dict, Tuple, List, Set

import gurobipy as gp
from gurobipy import GRB


def read_profiling_csv(path: str):
    """
    Returns:
      B: list of block ids
      S: list of tile ids
      data[(b,s)] = dict with keys: L, smem, reg, warp, occ, compat (0/1)
    """
    data: Dict[Tuple[str, str], Dict[str, float]] = {}
    blocks: Set[str] = set()
    tiles: Set[str] = set()

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["block_id", "tile_id", "latency", "smem", "reg", "warp", "occ_sm", "compat"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"profiling_csv missing columns: {missing}. Got: {reader.fieldnames}")

        for row in reader:
            b = row["block_id"].strip()
            s = row["tile_id"].strip()
            blocks.add(b)
            tiles.add(s)
            key = (b, s)
            data[key] = {
                "L": float(row["latency"]),
                "smem": float(row["smem"]),
                "reg": float(row["reg"]),
                "warp": float(row["warp"]),
                "occ": float(row["occ_sm"]),
                "compat": int(float(row["compat"])),  # allow "0.0/1.0"
            }

    return sorted(blocks), sorted(tiles), data


def read_tiles_envelope_csv(path: str):
    """
    Returns env[s] = dict {smem_env, reg_env, warp_env}
    """
    env = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["tile_id", "smem_env", "reg_env", "warp_env"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"tiles_csv missing columns: {missing}. Got: {reader.fieldnames}")

        for row in reader:
            s = row["tile_id"].strip()
            env[s] = {
                "smem_env": float(row["smem_env"]),
                "reg_env": float(row["reg_env"]),
                "warp_env": float(row["warp_env"]),
            }
    return env


def read_conflicts_csv(path: str):
    """
    Returns conflict[(s1,s2)] = 0/1 for s1 < s2 in lexicographic order.
    Missing pairs => 0.
    """
    conflict = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["tile_i", "tile_j", "conflict"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"conflicts_csv missing columns: {missing}. Got: {reader.fieldnames}")

        for row in reader:
            a = row["tile_i"].strip()
            b = row["tile_j"].strip()
            c = int(float(row["conflict"]))
            s1, s2 = (a, b) if a < b else (b, a)
            if s1 == s2:
                continue
            conflict[(s1, s2)] = 1 if c != 0 else 0
    return conflict


def build_envelopes_from_profiling(B, S, data):
    """
    If you don't have per-tile envelopes, use conservative max over blocks:
      smem_env[s] = max_b smem[b,s]
    similarly for reg/warp.
    """
    env = {s: {"smem_env": 0.0, "reg_env": 0.0, "warp_env": 0.0} for s in S}
    for s in S:
        sm = rg = wp = 0.0
        for b in B:
            d = data.get((b, s))
            if d is None or d["compat"] == 0:
                continue
            sm = max(sm, d["smem"])
            rg = max(rg, d["reg"])
            wp = max(wp, d["warp"])
        env[s]["smem_env"] = sm
        env[s]["reg_env"] = rg
        env[s]["warp_env"] = wp
    return env


def solve_ilp(
    B: List[str],
    S: List[str],
    R: int,
    data: Dict[Tuple[str, str], Dict[str, float]],
    env: Dict[str, Dict[str, float]],
    conflict: Dict[Tuple[str, str], int],
    kappa: int,
    smem_max: float,
    reg_max: float,
    warp_max: float,
    n_sm: int,
    time_limit_s: int = 0,
    mip_gap: float = 0.0,
    threads: int = 0,
    seed: int = 1,
):
    # Feasible tile set per block
    S_b = {b: [s for s in S if (b, s) in data and data[(b, s)]["compat"] == 1] for b in B}
    for b in B:
        if not S_b[b]:
            raise ValueError(f"No feasible tiles for block {b}. Check compat/profiling data.")

    m = gp.Model("ipdps_sparse_attention_ilp")

    # Params
    m.Params.OutputFlag = 1
    if time_limit_s > 0:
        m.Params.TimeLimit = time_limit_s
    if mip_gap > 0:
        m.Params.MIPGap = mip_gap
    if threads > 0:
        m.Params.Threads = threads
    m.Params.Seed = seed

    rounds = list(range(1, R + 1))

    # Variables
    x = {}  # (b,s,r) -> var
    for b in B:
        for s in S_b[b]:
            for r in rounds:
                x[(b, s, r)] = m.addVar(vtype=GRB.BINARY, name=f"x[{b},{s},{r}]")

    y = {(s, r): m.addVar(vtype=GRB.BINARY, name=f"y[{s},{r}]") for s in S for r in rounds}
    Tr = {r: m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Tr[{r}]") for r in rounds}
    T = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T")

    m.update()

    # Objective: minimize makespan T
    m.setObjective(T, GRB.MINIMIZE)

    # (C1) Each block assigned exactly once
    for b in B:
        m.addConstr(
            gp.quicksum(x[(b, s, r)] for s in S_b[b] for r in rounds) == 1,
            name=f"C1_assign[{b}]",
        )

    # (C2) Local feasibility + compat mask (and optional hard limits)
    # NOTE: If smem/reg/warp are per-SM limits, these constraints are redundant when compat already ensures feasibility.
    for b in B:
        for s in S_b[b]:
            d = data[(b, s)]
            for r in rounds:
                m.addConstr(d["smem"] * x[(b, s, r)] <= smem_max, name=f"C2_smem[{b},{s},{r}]")
                m.addConstr(d["reg"] * x[(b, s, r)] <= reg_max, name=f"C2_reg[{b},{s},{r}]")
                m.addConstr(d["warp"] * x[(b, s, r)] <= warp_max, name=f"C2_warp[{b},{s},{r}]")
                # compat is already enforced by restricting S_b, but keep for clarity:
                m.addConstr(x[(b, s, r)] <= d["compat"], name=f"C2_compat[{b},{s},{r}]")

    # (C3) activation logic + per-round tile cap kappa
    for r in rounds:
        for s in S:
            # if any block uses s in round r => y[s,r]=1
            m.addConstr(
                gp.quicksum(x[(b, s, r)] for b in B if (b, s, r) in x) <= 1e6 * y[(s, r)],
                name=f"C3_link_upper[{s},{r}]",
            )
        m.addConstr(gp.quicksum(y[(s, r)] for s in S) <= kappa, name=f"C3_kappa[{r}]")

    # (C4) Round time lower bound by selected block latency
    for r in rounds:
        for b in B:
            for s in S_b[b]:
                L = data[(b, s)]["L"]
                m.addConstr(Tr[r] >= L * x[(b, s, r)], name=f"C4_Tr[{b},{s},{r}]")

    # T >= Tr
    for r in rounds:
        m.addConstr(T >= Tr[r], name=f"T_ge_Tr[{r}]")

    # (C5) Round-level resource envelope (conservative)
    # Sum over activated tiles <= N_SM * per-SM limit
    for r in rounds:
        m.addConstr(
            gp.quicksum(env[s]["smem_env"] * y[(s, r)] for s in S) <= n_sm * smem_max,
            name=f"C5_smem_env[{r}]",
        )
        m.addConstr(
            gp.quicksum(env[s]["reg_env"] * y[(s, r)] for s in S) <= n_sm * reg_max,
            name=f"C5_reg_env[{r}]",
        )
        m.addConstr(
            gp.quicksum(env[s]["warp_env"] * y[(s, r)] for s in S) <= n_sm * warp_max,
            name=f"C5_warp_env[{r}]",
        )

    # (C6) SM occupancy (round capacity)
    for r in rounds:
        m.addConstr(
            gp.quicksum(data[(b, s)]["occ"] * x[(b, s, r)] for b in B for s in S_b[b]) <= n_sm,
            name=f"C6_occ[{r}]",
        )

    # (C7) Tile conflicts: if C[s,s']=1 then cannot both be active same round
    # y[s,r] + y[s',r] <= 1 for conflicting pairs
    for (s1, s2), c in conflict.items():
        if c != 1:
            continue
        for r in rounds:
            m.addConstr(y[(s1, r)] + y[(s2, r)] <= 1, name=f"C7_conf[{s1},{s2},{r}]")

    # Solve
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed with status {m.Status}")

    # Extract solution
    sol = {
        "status": int(m.Status),
        "objective_makespan": float(T.X),
        "rounds": {},
        "blocks": {},
    }

    # Per round: active tiles and Tr
    for r in rounds:
        active_tiles = [s for s in S if y[(s, r)].X > 0.5]
        sol["rounds"][str(r)] = {
            "Tr": float(Tr[r].X),
            "active_tiles": active_tiles,
        }

    # Per block: assigned (s,r)
    for b in B:
        chosen = None
        for r in rounds:
            for s in S_b[b]:
                v = x[(b, s, r)]
                if v.X > 0.5:
                    chosen = (s, r)
                    break
            if chosen:
                break
        if chosen is None:
            # can happen if TIME_LIMIT without integer feasible? rare but handle
            sol["blocks"][b] = {"round": None, "tile": None}
        else:
            s, r = chosen
            sol["blocks"][b] = {"round": int(r), "tile": s}

    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiling_csv", required=True, help="profiling table CSV")
    ap.add_argument("--tiles_csv", default="", help="optional per-tile envelope CSV")
    ap.add_argument("--conflicts_csv", default="", help="optional tile conflict CSV")
    ap.add_argument("--out_json", default="schedule.json", help="output schedule JSON")

    ap.add_argument("--rounds", type=int, required=True, help="R_max")
    ap.add_argument("--kappa", type=int, default=4, help="max active tile types per round")
    ap.add_argument("--n_sm", type=int, default=108, help="number of SMs (A100=108)")
    ap.add_argument("--smem_max", type=float, default=163840, help="bytes per SM (A100=163840)")
    ap.add_argument("--reg_max", type=float, default=65536, help="registers per SM (architecture dependent)")
    ap.add_argument("--warp_max", type=float, default=64, help="max warps per SM (typical)")
    ap.add_argument("--time_limit", type=int, default=0, help="seconds (0 = no limit)")
    ap.add_argument("--mip_gap", type=float, default=0.0, help="relative MIP gap, e.g., 0.01")
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0=auto)")
    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    B, S, data = read_profiling_csv(args.profiling_csv)

    if args.tiles_csv:
        env = read_tiles_envelope_csv(args.tiles_csv)
        # ensure all tiles exist
        for s in S:
            if s not in env:
                raise ValueError(f"tiles_csv missing envelope for tile {s}")
    else:
        env = build_envelopes_from_profiling(B, S, data)

    conflict = {}
    if args.conflicts_csv:
        conflict = read_conflicts_csv(args.conflicts_csv)

    sol = solve_ilp(
        B=B,
        S=S,
        R=args.rounds,
        data=data,
        env=env,
        conflict=conflict,
        kappa=args.kappa,
        smem_max=args.smem_max,
        reg_max=args.reg_max,
        warp_max=args.warp_max,
        n_sm=args.n_sm,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
    )

    with open(args.out_json, "w") as f:
        json.dump(sol, f, indent=2)

    print(f"[OK] wrote {args.out_json}")
    print(f"makespan T = {sol['objective_makespan']:.6f}")


if __name__ == "__main__":
    main()
