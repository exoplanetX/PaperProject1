from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from 第一阶段环境搭建_fixed import DataGenerator, Environment
from 第二阶段算法搭建_fixed import GreedyPolicy, BalancePolicy, RBAPolicy
from offline_opt_fixed import solve_paper_lp_benchmark


def run_one_trial(n: int, c: int, dist_type: str, kappa: float, seed: int, include_bursty: bool = True):
    generator = DataGenerator(
        n=n,
        c=c,
        dist_type=dist_type,
        kappa=kappa,
        seed=seed,
        include_bursty=include_bursty,
    )

    policies = {
        "Greedy": GreedyPolicy(),
        "Balance": BalancePolicy(c=c),
        "RBA": RBAPolicy(c=c),
    }

    results = {}
    for name, policy in policies.items():
        env = Environment(generator)
        total_reward = 0.0

        while not env.done():
            state = env.observe()
            action = policy.decide(state["edges"], env.available_inventory)
            log = env.step(action)
            if log["allocated"] is not None:
                total_reward += 1.0

        results[name] = total_reward

    return results


def run_experiment(
    n: int = 5,
    capacities = (5, 15, 25),
    dist_types = ("two_point", "exponential", "weibull"),
    kappa: float = 1.0,
    num_instances: int = 100,
    include_bursty: bool = True,
):
    final_results = {}

    for dist in dist_types:
        final_results[dist] = {}
        for c in capacities:
            generator_for_lp = DataGenerator(
                n=n,
                c=c,
                dist_type=dist,
                kappa=kappa,
                seed=0,
                include_bursty=include_bursty,
            )
            lp_value = solve_paper_lp_benchmark(generator_for_lp)

            raw = {"Greedy": [], "Balance": [], "RBA": []}
            for seed in range(num_instances):
                rewards = run_one_trial(
                    n=n,
                    c=c,
                    dist_type=dist,
                    kappa=kappa,
                    seed=seed,
                    include_bursty=include_bursty,
                )
                for algo, reward in rewards.items():
                    raw[algo].append(reward / lp_value)

            final_results[dist][c] = {
                algo: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
                for algo, vals in raw.items()
            }

    return final_results


def plot_results(results, title_suffix: str = ""):
    for dist, dist_results in results.items():
        plt.figure(figsize=(8, 5.5))
        capacities = sorted(dist_results.keys())
        for algo in ["Greedy", "Balance", "RBA"]:
            means = np.array([dist_results[c][algo]["mean"] for c in capacities])
            stds = np.array([dist_results[c][algo]["std"] for c in capacities])
            plt.plot(capacities, means, marker="o", label=algo)
            plt.fill_between(capacities, means - stds, means + stds, alpha=0.15)

        plt.axhline(1 - np.exp(-1), linestyle="--", linewidth=1.0, label="1 - 1/e")
        plt.ylim(0.45, 1.02)
        plt.xlabel("Capacity c")
        plt.ylabel("Average ratio to LP benchmark")
        plt.title(f"{dist}{title_suffix}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    results = run_experiment(n=5, capacities=(5, 15, 25), kappa=1.0, num_instances=20, include_bursty=True)
    print(results)
    plot_results(results, title_suffix=" (paper-style setting)")
