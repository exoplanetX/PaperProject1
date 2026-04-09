from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Arrival:
    time: int
    phase: int
    is_burst: bool
    arrival_type: int  # 1..2n in the paper
    edges: List[int]   # 0-based resource indices


class DataGenerator:
    """
    Paper-aligned instance generator for Section 6 of Goyal et al. (2025).

    - n resources, each with capacity c
    - n phases
    - each phase has c burst arrivals at the same time, followed by c normal arrivals
    - total arrivals = 2*c*n when include_bursty=True
    - if include_bursty=False, only the normal arrivals are kept (Table 4 setting)
    """

    def __init__(
        self,
        n: int,
        c: int,
        dist_type: str = "exponential",
        kappa: float = 1.0,
        seed: Optional[int] = None,
        include_bursty: bool = True,
    ) -> None:
        self.n = n
        self.c = c
        self.dist_type = dist_type
        self.kappa = kappa
        self.include_bursty = include_bursty
        self.rng = np.random.default_rng(seed)

        self.arrivals: List[Arrival] = self._build_arrivals()
        self.T = len(self.arrivals)
        self.arrival_times = [a.time for a in self.arrivals]
        self.customer_edges = [a.edges for a in self.arrivals]
        self.edge_probabilities = self._build_edge_probabilities()

        # Common-random-number duration pools: one i.i.d. list per (resource, unit).
        self.duration_pools: Dict[Tuple[int, int], List[int]] = {
            (i, u): [self.sample_duration() for _ in range(self.T)]
            for i in range(self.n)
            for u in range(1, self.c + 1)
        }

    def _normal_type_distribution(self, phase: int) -> np.ndarray:
        """Probability over normal arrival types 1..n in a given phase (1-based phase)."""
        peak_type = self.n - phase + 1
        weights = np.array(
            [np.exp(-self.kappa * abs(t - peak_type)) for t in range(1, self.n + 1)],
            dtype=float,
        )
        return weights / weights.sum()

    def _type_to_edges(self, arrival_type: int) -> List[int]:
        """Map paper arrival type to 0-based feasible resources."""
        if 1 <= arrival_type <= self.n:
            # Type i has edge to every resource in [i] (1-based).
            return list(range(arrival_type))
        if self.n + 1 <= arrival_type <= 2 * self.n:
            # Type n+j has edge only to resource j.
            return [arrival_type - self.n - 1]
        raise ValueError("arrival_type out of range")

    def _build_arrivals(self) -> List[Arrival]:
        arrivals: List[Arrival] = []
        for phase in range(1, self.n + 1):
            start_time = (self.c + 1) * (phase - 1) + 1

            if self.include_bursty:
                burst_type = 2 * self.n - phase + 1
                burst_edges = self._type_to_edges(burst_type)
                for _ in range(self.c):
                    arrivals.append(
                        Arrival(
                            time=start_time,
                            phase=phase,
                            is_burst=True,
                            arrival_type=burst_type,
                            edges=burst_edges,
                        )
                    )

            probs = self._normal_type_distribution(phase)
            for dt in range(1, self.c + 1):
                arrival_type = int(self.rng.choice(np.arange(1, self.n + 1), p=probs))
                arrivals.append(
                    Arrival(
                        time=start_time + dt,
                        phase=phase,
                        is_burst=False,
                        arrival_type=arrival_type,
                        edges=self._type_to_edges(arrival_type),
                    )
                )
        return arrivals

    def _build_edge_probabilities(self) -> np.ndarray:
        """p_{i,t} used by the paper LP benchmark."""
        p = np.zeros((self.n, self.T), dtype=float)
        idx = 0
        for phase in range(1, self.n + 1):
            peak_type = self.n - phase + 1
            weights = np.array(
                [np.exp(-self.kappa * abs(t - peak_type)) for t in range(1, self.n + 1)],
                dtype=float,
            )
            denom = weights.sum()

            if self.include_bursty:
                burst_resource = self.n - phase  # 0-based
                for _ in range(self.c):
                    p[burst_resource, idx] = 1.0
                    idx += 1

            # normal arrivals
            suffix_sums = np.cumsum(weights[::-1])[::-1]
            for _ in range(self.c):
                for i in range(self.n):
                    # resource i is feasible iff sampled type >= i+1
                    p[i, idx] = suffix_sums[i] / denom
                idx += 1
        return p

    def sample_duration(self) -> int:
        if self.dist_type == "exponential":
            return max(1, int(np.ceil(self.rng.exponential(self.c))))
        if self.dist_type == "weibull":
            shape = 0.5
            scale = self.c / 2.0
            return max(1, int(np.ceil(self.rng.weibull(shape) * scale)))
        if self.dist_type == "two_point":
            return 1 if self.rng.random() < 0.5 else self.c
        raise ValueError("Unknown distribution")


class Environment:
    """Event-driven reusable-resource simulator with possibly repeated arrival times."""

    def __init__(self, generator: DataGenerator):
        self.gen = generator
        self.ptr = 0
        self.current_arrival: Optional[Arrival] = None
        self.prepared = False

        # Units are indexed 1..c; larger index = higher rank.
        self.available_inventory: Dict[int, List[int]] = {
            i: list(range(1, self.gen.c + 1)) for i in range(self.gen.n)
        }
        self.return_queue: List[Tuple[int, int, int]] = []  # (return_time, resource, unit)
        self.duration_ptr: Dict[Tuple[int, int], int] = {
            (i, u): 0 for i in range(self.gen.n) for u in range(1, self.gen.c + 1)
        }

    def done(self) -> bool:
        return self.ptr >= self.gen.T

    def observe(self) -> Dict:
        if self.done():
            raise IndexError("No more arrivals.")
        if self.prepared:
            return self._state_dict([])

        arrival = self.gen.arrivals[self.ptr]
        returned = []
        while self.return_queue and self.return_queue[0][0] <= arrival.time:
            return_time, resource_id, unit_id = heapq.heappop(self.return_queue)
            self.available_inventory[resource_id].append(unit_id)
            self.available_inventory[resource_id].sort()
            returned.append((return_time, resource_id, unit_id))

        self.current_arrival = arrival
        self.prepared = True
        return self._state_dict(returned)

    def _state_dict(self, returned: List[Tuple[int, int, int]]) -> Dict:
        assert self.current_arrival is not None or self.ptr == 0 or self.done() or self.prepared
        arrival = self.gen.arrivals[self.ptr]
        return {
            "index": self.ptr,
            "time": arrival.time,
            "phase": arrival.phase,
            "is_burst": arrival.is_burst,
            "arrival_type": arrival.arrival_type,
            "edges": arrival.edges,
            "returned": returned,
            "inventory": {k: v.copy() for k, v in self.available_inventory.items()},
        }

    def step(self, action: Optional[Tuple[int, int]]) -> Dict:
        if not self.prepared:
            self.observe()
        assert self.current_arrival is not None
        arrival = self.current_arrival

        allocated = None
        duration = None
        return_time = None

        if action is not None:
            resource_id, unit_id = action
            if resource_id in arrival.edges and unit_id in self.available_inventory[resource_id]:
                self.available_inventory[resource_id].remove(unit_id)
                allocated = (resource_id, unit_id)
                key = (resource_id, unit_id)
                draw_idx = self.duration_ptr[key]
                duration = self.gen.duration_pools[key][draw_idx]
                self.duration_ptr[key] += 1
                return_time = arrival.time + duration
                heapq.heappush(self.return_queue, (return_time, resource_id, unit_id))

        log = {
            "index": self.ptr,
            "time": arrival.time,
            "phase": arrival.phase,
            "is_burst": arrival.is_burst,
            "arrival_type": arrival.arrival_type,
            "edges": arrival.edges,
            "allocated": allocated,
            "duration": duration,
            "return_time": return_time,
            "inventory_after": {k: v.copy() for k, v in self.available_inventory.items()},
        }

        self.ptr += 1
        self.current_arrival = None
        self.prepared = False
        return log
