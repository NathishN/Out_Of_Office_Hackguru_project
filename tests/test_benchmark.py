import unittest
from src.benchmark import BenchmarkRunner

class TestBenchmark(unittest.TestCase):
    def test_benchmark_metrics_completeness(self):
        """Verify that benchmark records all metrics required by functional specs."""
        runner = BenchmarkRunner(num_runs=2, num_customers=6, num_vehicles=2, capacity=50)
        df, summary = runner.run_benchmark(inject_disruption=False, compare_modes=False)
        
        expected_columns = [
            "Seed", "Algorithm", "Disrupted", "Total Cost", "Runtime (ms)",
            "Feasible", "Deliveries Completed", "Constraint Violations",
            "Unassigned", "Vehicles Used"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
            
        self.assertEqual(len(df), 4)  # 2 seeds * 2 algorithms
        
        # Verify summary keys
        expected_summary_keys = [
            "Baseline Mean Cost", "Baseline Best Cost", "Baseline Worst Cost",
            "SA Mean Cost", "SA Best Cost", "SA Worst Cost",
            "Baseline Mean Runtime", "SA Mean Runtime",
            "Baseline Feasibility Rate", "SA Feasibility Rate"
        ]
        for key in expected_summary_keys:
            self.assertIn(key, summary)

    def test_benchmark_fair_comparison(self):
        """Verify that both algorithms are evaluated on the identical seeds and instances."""
        runner = BenchmarkRunner(num_runs=3, num_customers=8, num_vehicles=2, capacity=50)
        df, _ = runner.run_benchmark(inject_disruption=False, compare_modes=False)
        
        base_seeds = df[df["Algorithm"] == "Nearest Neighbor + 2-opt"]["Seed"].tolist()
        sa_seeds = df[df["Algorithm"] == "Simulated Annealing"]["Seed"].tolist()
        self.assertEqual(base_seeds, sa_seeds)
        self.assertEqual(base_seeds, [42, 43, 44])

    def test_benchmark_comparative_modes(self):
        """Verify comparative benchmark evaluates both normal and disrupted scenarios."""
        runner = BenchmarkRunner(num_runs=2, num_customers=6, num_vehicles=2, capacity=60)
        df, _ = runner.run_benchmark(compare_modes=True)
        
        # 2 seeds * 2 modes (normal, disrupted) * 2 algorithms = 8 rows
        self.assertEqual(len(df), 8)
        self.assertIn(True, df["Disrupted"].values)
        self.assertIn(False, df["Disrupted"].values)

if __name__ == '__main__':
    unittest.main()
