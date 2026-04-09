from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from 第一阶段环境搭建_fixed import DataGenerator


def survival_prob(delta: float, dist_type: str, c: int) -> float:
    """Return 1 - F(delta) for the paper distributions."""
    if delta < 0:
        return 1.0
    if dist_type == "exponential":
        return float(np.exp(-delta / c))
    if dist_type == "weibull":
        return float(np.exp(-((2.0 * delta) / c) ** 0.5))
    if dist_type == "two_point":
        if delta < 1:
            return 1.0
        if delta < c:
            return 0.5
        return 0.0
    raise ValueError("Unknown distribution")


def solve_paper_lp_benchmark(generator: DataGenerator) -> float:
    """
    Solve Appendix-I LP benchmark:
        max sum_{i,t} y_{i,t}
        s.t. sum_{t<=tau} [1-F(a_tau-a_t)] y_{i,t} <= c
             sum_i y_{i,t} <= 1
             0 <= y_{i,t} <= p_{i,t}
    """
    n = generator.n
    T = generator.T
    c = generator.c
    arrival_times = np.array(generator.arrival_times, dtype=float)
    p = generator.edge_probabilities

    num_vars = n * T
    c_obj = -np.ones(num_vars, dtype=float)  # maximize -> minimize negative objective
    bounds = [(0.0, float(p[i, t])) for i in range(n) for t in range(T)]

    # Capacity constraints
    cap_rows = []
    cap_cols = []
    cap_data = []
    b_cap = []

    for i in range(n):
        for tau in range(T):
            row_id = len(b_cap)
            b_cap.append(float(c))
            a_tau = arrival_times[tau]
            for t in range(tau + 1):
                coeff = survival_prob(a_tau - arrival_times[t], generator.dist_type, c)
                if coeff <= 1e-14:
                    continue
                col_id = i * T + t
                cap_rows.append(row_id)
                cap_cols.append(col_id)
                cap_data.append(coeff)

    A_cap = coo_matrix((cap_data, (cap_rows, cap_cols)), shape=(len(b_cap), num_vars))

    # One-match-per-arrival constraints
    match_rows = []
    match_cols = []
    match_data = []
    b_match = np.ones(T, dtype=float)
    for t in range(T):
        for i in range(n):
            col_id = i * T + t
            match_rows.append(t)
            match_cols.append(col_id)
            match_data.append(1.0)
    A_match = coo_matrix((match_data, (match_rows, match_cols)), shape=(T, num_vars))

    A_ub = vstack([A_cap, A_match], format="csr")
    b_ub = np.concatenate([np.array(b_cap, dtype=float), b_match])

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP benchmark failed: {res.message}")
    return float(-res.fun)
