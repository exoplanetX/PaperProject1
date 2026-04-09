from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class GreedyPolicy:
    """Paper-consistent greedy with deterministic tie-breaking."""

    def decide(
        self,
        edges: List[int],
        inventory: Dict[int, List[int]],
    ) -> Optional[Tuple[int, int]]:
        feasible = [i for i in sorted(edges) if inventory[i]]
        if not feasible:
            return None
        resource = feasible[0]
        unit = min(inventory[resource])
        return resource, unit


class BalancePolicy:
    """Balance: argmax_i 1 - exp(-y_i/c), ties -> lowest resource index."""

    def __init__(self, c: int):
        self.c = c

    def decide(
        self,
        edges: List[int],
        inventory: Dict[int, List[int]],
    ) -> Optional[Tuple[int, int]]:
        best_resource = None
        best_score = -1.0

        for i in sorted(edges):
            y_i = len(inventory[i])
            if y_i == 0:
                continue
            score = 1.0 - math.exp(-y_i / self.c)
            if score > best_score + 1e-12:
                best_score = score
                best_resource = i

        if best_resource is None:
            return None
        return best_resource, min(inventory[best_resource])


class RBAPolicy:
    """RBA: argmax_i 1 - exp(-z_i(t)/c), where z_i(t) is highest ranked available unit."""

    def __init__(self, c: int):
        self.c = c

    def decide(
        self,
        edges: List[int],
        inventory: Dict[int, List[int]],
    ) -> Optional[Tuple[int, int]]:
        best_resource = None
        best_unit = None
        best_score = -1.0

        for i in sorted(edges):
            if not inventory[i]:
                continue
            z_i = max(inventory[i])
            score = 1.0 - math.exp(-z_i / self.c)
            if score > best_score + 1e-12:
                best_score = score
                best_resource = i
                best_unit = z_i

        if best_resource is None:
            return None
        return best_resource, best_unit
