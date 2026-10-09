import unittest
import copy
from src.data_generator import generate_cvrp_instance
from src.baseline_optimizer import BaselineOptimizer
from src.routing_model import Solution, Route

class TestBaselineOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = BaselineOptimizer(use_two_opt=True)

    def test_solve_cvrp_instance(self):
        instance = generate_cvrp_instance(num_customers=10, num_vehicles=3, vehicle_capacity=60, seed=42)
        solution = self.optimizer.solve(instance)
        
        self.assertTrue(solution.is_feasible)
        self.assertEqual(len(solution.unassigned_nodes), 0)
        self.assertGreater(solution.total_cost, 0.0)
        self.assertGreater(solution.execution_time_ms, 0.0)

    def test_few_delivery_locations(self):
        """Test with small instance (3 customer nodes)."""
        instance = generate_cvrp_instance(num_customers=3, num_vehicles=2, vehicle_capacity=50, seed=42)
        solution = self.optimizer.solve(instance)
        
        self.assertTrue(solution.is_feasible)
        self.assertEqual(len(solution.unassigned_nodes), 0)
        visited = [nid for r in solution.routes for nid in r.node_ids if nid != 0]
        self.assertEqual(sorted(visited), [1, 2, 3])

    def test_multiple_vehicles(self):
        """Test routing with multiple vehicles (5 vehicles)."""
        instance = generate_cvrp_instance(num_customers=15, num_vehicles=5, vehicle_capacity=50, seed=42)
        solution = self.optimizer.solve(instance)
        
        self.assertTrue(solution.is_feasible)
        active_routes = [r for r in solution.routes if len(r.node_ids) > 2]
        self.assertGreaterEqual(len(active_routes), 1)
        self.assertLessEqual(len(active_routes), 5)

    def test_all_deliveries_served(self):
        """Validate that 100% of customers are served when capacity is sufficient."""
        instance = generate_cvrp_instance(num_customers=12, num_vehicles=4, vehicle_capacity=80, seed=10)
        solution = self.optimizer.solve(instance)
        
        all_visited = set()
        for route in solution.routes:
            for nid in route.node_ids:
                if nid != 0:
                    all_visited.add(nid)
        self.assertEqual(len(all_visited), 12)
        self.assertEqual(len(solution.unassigned_nodes), 0)
        self.assertTrue(solution.is_feasible)

    def test_insufficient_capacity_handling(self):
        """
        When total vehicle capacity is strictly smaller than required customer demand,
        the optimizer must report unassigned nodes and mark the solution infeasible.
        """
        # 1 vehicle with capacity 10, but each customer needs 10-20
        instance = generate_cvrp_instance(
            num_customers=8,
            num_vehicles=1,
            vehicle_capacity=15,
            min_demand=10,
            max_demand=15,
            seed=42
        )
        solution = self.optimizer.solve(instance)
        
        self.assertFalse(solution.is_feasible)
        self.assertGreater(len(solution.unassigned_nodes), 0)

    def test_route_cost_calculation(self):
        """Verify that total cost matches exact sum of individual edge distances."""
        instance = generate_cvrp_instance(num_customers=6, num_vehicles=2, vehicle_capacity=60, seed=42)
        solution = self.optimizer.solve(instance)
        
        computed_total = 0.0
        for route in solution.routes:
            route_cost = 0.0
            for i in range(len(route.node_ids) - 1):
                u, v = route.node_ids[i], route.node_ids[i + 1]
                route_cost += instance.cost_matrix[u, v]
            self.assertAlmostEqual(route.total_cost, route_cost, places=4)
            computed_total += route_cost

        self.assertAlmostEqual(solution.total_cost, computed_total, places=4)

    def test_duplicate_or_missing_deliveries_detection(self):
        """
        Verify that validate_feasibility correctly identifies solutions
        with duplicate customer visits or missing deliveries.
        """
        instance = generate_cvrp_instance(num_customers=4, num_vehicles=2, vehicle_capacity=50, seed=42)
        valid_solution = self.optimizer.solve(instance)
        self.assertTrue(valid_solution.is_feasible)

        # 1. Solution with missing delivery
        missing_sol = copy.deepcopy(valid_solution)
        # Drop a customer from route
        for r in missing_sol.routes:
            if len(r.node_ids) > 3:
                r.node_ids.pop(1)
                break
        self.assertFalse(missing_sol.validate_feasibility(instance.vehicle_capacity, instance.num_customers))

        # 2. Solution with duplicate delivery
        dup_sol = copy.deepcopy(valid_solution)
        # Append customer 1 again to the first route
        dup_sol.routes[0].node_ids.insert(1, 1)
        self.assertFalse(dup_sol.validate_feasibility(instance.vehicle_capacity, instance.num_customers))

    def test_reproducibility(self):
        """Baseline algorithm is deterministic given identical problem instances."""
        inst1 = generate_cvrp_instance(num_customers=10, seed=77)
        inst2 = generate_cvrp_instance(num_customers=10, seed=77)
        sol1 = self.optimizer.solve(inst1)
        sol2 = self.optimizer.solve(inst2)
        
        self.assertAlmostEqual(sol1.total_cost, sol2.total_cost, places=5)
        for r1, r2 in zip(sol1.routes, sol2.routes):
            self.assertEqual(r1.node_ids, r2.node_ids)

if __name__ == '__main__':
    unittest.main()
