import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.data_generator import generate_cvrp_instance
from src.baseline_optimizer import BaselineOptimizer
from src.simulated_annealing import SimulatedAnnealingOptimizer
from src.disruption_simulator import DisruptionSimulator

class BenchmarkRunner:
    """
    Executes reproducible benchmark experiments comparing baseline and quantum-inspired optimization algorithms.
    """

    def __init__(self, num_runs: int = 10, num_customers: int = 15, num_vehicles: int = 3, capacity: int = 50):
        self.num_runs = num_runs
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity

    def run_benchmark(self, inject_disruption: bool = False, compare_modes: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs benchmark experiments across seeds.
        If compare_modes is True, runs each seed under both Normal and Disrupted scenarios.
        """
        results = []
        baseline_solver = BaselineOptimizer(use_two_opt=True)
        modes = [False, True] if compare_modes else [inject_disruption]

        for seed in range(42, 42 + self.num_runs):
            for disrupted_flag in modes:
                # Generate identical instance
                instance = generate_cvrp_instance(
                    num_customers=self.num_customers,
                    num_vehicles=self.num_vehicles,
                    vehicle_capacity=self.capacity,
                    seed=seed
                )

                if disrupted_flag:
                    # Block edge between depot (0) and customer 1
                    instance = DisruptionSimulator.apply_edge_closure(instance, 0, 1)

                # Solve Baseline
                sol_base = baseline_solver.solve(instance)
                results.append({
                    "Seed": seed,
                    "Algorithm": "Nearest Neighbor + 2-opt",
                    "Disrupted": disrupted_flag,
                    "Total Cost": round(sol_base.total_cost, 2),
                    "Runtime (ms)": round(sol_base.execution_time_ms, 2),
                    "Feasible": sol_base.is_feasible,
                    "Deliveries Completed": instance.num_customers - len(sol_base.unassigned_nodes),
                    "Constraint Violations": 0 if sol_base.is_feasible else 1,
                    "Unassigned": len(sol_base.unassigned_nodes),
                    "Vehicles Used": len([r for r in sol_base.routes if len(r.node_ids) > 2])
                })

                # Solve Simulated Annealing
                sa_solver = SimulatedAnnealingOptimizer(seed=seed, max_iterations=3000)
                sol_sa = sa_solver.solve(instance)
                results.append({
                    "Seed": seed,
                    "Algorithm": "Simulated Annealing",
                    "Disrupted": disrupted_flag,
                    "Total Cost": round(sol_sa.total_cost, 2),
                    "Runtime (ms)": round(sol_sa.execution_time_ms, 2),
                    "Feasible": sol_sa.is_feasible,
                    "Deliveries Completed": instance.num_customers - len(sol_sa.unassigned_nodes),
                    "Constraint Violations": 0 if sol_sa.is_feasible else 1,
                    "Unassigned": len(sol_sa.unassigned_nodes),
                    "Vehicles Used": len([r for r in sol_sa.routes if len(r.node_ids) > 2])
                })

        df = pd.DataFrame(results)

        base_df = df[df["Algorithm"] == "Nearest Neighbor + 2-opt"]
        sa_df = df[df["Algorithm"] == "Simulated Annealing"]

        summary = {
            "Baseline Mean Cost": float(base_df["Total Cost"].mean()),
            "Baseline Best Cost": float(base_df["Total Cost"].min()),
            "Baseline Worst Cost": float(base_df["Total Cost"].max()),
            "SA Mean Cost": float(sa_df["Total Cost"].mean()),
            "SA Best Cost": float(sa_df["Total Cost"].min()),
            "SA Worst Cost": float(sa_df["Total Cost"].max()),
            "Baseline Mean Runtime": float(base_df["Runtime (ms)"].mean()),
            "SA Mean Runtime": float(sa_df["Runtime (ms)"].mean()),
            "Baseline Feasibility Rate": float(base_df["Feasible"].mean() * 100),
            "SA Feasibility Rate": float(sa_df["Feasible"].mean() * 100),
        }

        return df, summary
