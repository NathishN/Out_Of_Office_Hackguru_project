import unittest
from src.data_generator import generate_cvrp_instance
from src.simulated_annealing import SimulatedAnnealingOptimizer

class TestSimulatedAnnealing(unittest.TestCase):
    def test_solve_simulated_annealing(self):
        instance = generate_cvrp_instance(num_customers=8, num_vehicles=3, vehicle_capacity=60, seed=42)
        optimizer = SimulatedAnnealingOptimizer(max_iterations=1000, seed=42)
        solution = optimizer.solve(instance)
        
        self.assertGreater(solution.total_cost, 0.0)
        self.assertGreater(solution.execution_time_ms, 0.0)
        self.assertTrue(solution.is_feasible)
        self.assertEqual(len(solution.unassigned_nodes), 0)

    def test_reproducibility_with_seed(self):
        """Simulated Annealing is reproducible when fixed seed is provided."""
        instance = generate_cvrp_instance(num_customers=8, num_vehicles=2, vehicle_capacity=60, seed=42)
        opt1 = SimulatedAnnealingOptimizer(max_iterations=800, seed=123)
        opt2 = SimulatedAnnealingOptimizer(max_iterations=800, seed=123)
        
        sol1 = opt1.solve(instance)
        sol2 = opt2.solve(instance)
        self.assertAlmostEqual(sol1.total_cost, sol2.total_cost, places=4)
        for r1, r2 in zip(sol1.routes, sol2.routes):
            self.assertEqual(r1.node_ids, r2.node_ids)

    def test_few_delivery_locations(self):
        """Test with small instance (3 customer nodes)."""
        instance = generate_cvrp_instance(num_customers=3, num_vehicles=2, vehicle_capacity=50, seed=42)
        optimizer = SimulatedAnnealingOptimizer(max_iterations=500, seed=42)
        solution = optimizer.solve(instance)
        
        self.assertTrue(solution.is_feasible)
        visited = [nid for r in solution.routes for nid in r.node_ids if nid != 0]
        self.assertEqual(sorted(visited), [1, 2, 3])

    def test_multiple_vehicles(self):
        """Test with multiple vehicles (4 vehicles)."""
        instance = generate_cvrp_instance(num_customers=12, num_vehicles=4, vehicle_capacity=60, seed=42)
        optimizer = SimulatedAnnealingOptimizer(max_iterations=800, seed=42)
        solution = optimizer.solve(instance)
        
        self.assertEqual(len(solution.routes), 4)
        self.assertGreater(solution.total_cost, 0.0)

    def test_insufficient_capacity_flags_infeasible(self):
        """
        When total demand substantially exceeds vehicle capacity,
        the solution must be flagged as infeasible due to capacity violation.
        """
        # 1 vehicle with capacity 10, but each customer requires 15
        instance = generate_cvrp_instance(
            num_customers=5,
            num_vehicles=1,
            vehicle_capacity=10,
            min_demand=15,
            max_demand=15,
            seed=42
        )
        optimizer = SimulatedAnnealingOptimizer(max_iterations=400, seed=42)
        solution = optimizer.solve(instance)
        
        self.assertFalse(solution.is_feasible)

    def test_all_deliveries_served(self):
        """Verify that all customer nodes are visited when capacity is sufficient."""
        instance = generate_cvrp_instance(num_customers=10, num_vehicles=3, vehicle_capacity=70, seed=55)
        optimizer = SimulatedAnnealingOptimizer(max_iterations=1000, seed=55)
        solution = optimizer.solve(instance)
        
        visited = set()
        for r in solution.routes:
            for nid in r.node_ids:
                if nid != 0:
                    visited.add(nid)
        self.assertEqual(len(visited), 10)

if __name__ == '__main__':
    unittest.main()
