import unittest
from src.data_generator import generate_cvrp_instance
from src.baseline_optimizer import BaselineOptimizer
from src.simulated_annealing import SimulatedAnnealingOptimizer
from src.disruption_simulator import DisruptionSimulator, BLOCKED_COST

class TestDisruptionSimulator(unittest.TestCase):
    def test_edge_closure_disruption(self):
        instance = generate_cvrp_instance(num_customers=8, num_vehicles=2, vehicle_capacity=60, seed=42)
        disrupted = DisruptionSimulator.apply_edge_closure(instance, 0, 1)
        
        self.assertEqual(disrupted.cost_matrix[0, 1], BLOCKED_COST)
        self.assertEqual(disrupted.cost_matrix[1, 0], BLOCKED_COST)
        
        # Verify baseline avoids blocked edge
        solver = BaselineOptimizer(use_two_opt=True)
        sol = solver.solve(disrupted)
        
        for route in sol.routes:
            nodes = route.node_ids
            for i in range(len(nodes) - 1):
                self.assertFalse((nodes[i] == 0 and nodes[i+1] == 1) or (nodes[i] == 1 and nodes[i+1] == 0))

    def test_traffic_penalty(self):
        """Verify traffic penalty increases edge cost by the multiplier."""
        instance = generate_cvrp_instance(num_customers=6, num_vehicles=2, vehicle_capacity=50, seed=42)
        original_cost = instance.cost_matrix[0, 1]
        
        disrupted = DisruptionSimulator.apply_traffic_penalty(instance, 0, 1, multiplier=4.0)
        self.assertAlmostEqual(disrupted.cost_matrix[0, 1], original_cost * 4.0, places=4)
        self.assertAlmostEqual(disrupted.cost_matrix[1, 0], original_cost * 4.0, places=4)

    def test_frequently_used_edge_disruption(self):
        """
        Identify an edge actively traversed in the initial baseline route,
        block it, and verify that replanning completely eliminates that edge.
        """
        instance = generate_cvrp_instance(num_customers=10, num_vehicles=3, vehicle_capacity=60, seed=42)
        solver = BaselineOptimizer(use_two_opt=True)
        initial_sol = solver.solve(instance)
        
        # Pick first active edge
        target_edge = None
        for route in initial_sol.routes:
            if len(route.node_ids) > 2:
                target_edge = (route.node_ids[0], route.node_ids[1])
                break
        self.assertIsNotNone(target_edge)
        
        # Check affected routes identification
        affected = DisruptionSimulator.check_affected_routes(initial_sol, [target_edge])
        self.assertGreaterEqual(len(affected), 1)

        # Apply closure to that active edge
        u, v = target_edge
        disrupted = DisruptionSimulator.apply_edge_closure(instance, u, v)

        # Replan
        replanned_sol = solver.solve(disrupted)
        self.assertTrue(replanned_sol.is_feasible)

        # Verify new routes do not use the blocked edge
        for route in replanned_sol.routes:
            nodes = route.node_ids
            for i in range(len(nodes) - 1):
                self.assertFalse((nodes[i] == u and nodes[i+1] == v) or (nodes[i] == v and nodes[i+1] == u))

    def test_vehicle_breakdown_reassignment(self):
        """
        Simulate a vehicle breakdown by reducing fleet size,
        and ensure remaining vehicles cover the demand if feasible.
        """
        instance = generate_cvrp_instance(num_customers=6, num_vehicles=3, vehicle_capacity=70, seed=42)
        disrupted = DisruptionSimulator.apply_vehicle_breakdown(instance, breakdown_vehicle_id=0)
        
        self.assertEqual(disrupted.num_vehicles, 2)
        
        solver = BaselineOptimizer(use_two_opt=True)
        replanned_sol = solver.solve(disrupted)
        
        self.assertTrue(replanned_sol.is_feasible)
        self.assertEqual(len(replanned_sol.unassigned_nodes), 0)
        self.assertEqual(len(replanned_sol.routes), 2)

    def test_unavailable_edge_validator_flags_infeasibility(self):
        """
        Verify that validate_feasibility with cost_matrix flags any route
        containing an edge with cost >= 1e5 as infeasible.
        """
        instance = generate_cvrp_instance(num_customers=5, num_vehicles=2, vehicle_capacity=60, seed=42)
        disrupted = DisruptionSimulator.apply_edge_closure(instance, 0, 1)
        
        # Solve on normal instance, where edge 0-1 might be used
        solver = BaselineOptimizer(use_two_opt=True)
        normal_sol = solver.solve(instance)

        # Force a route to use the blocked edge
        normal_sol.routes[0].node_ids = [0, 1, 0]
        normal_sol.compute_total_cost(disrupted.cost_matrix)

        # Feasibility check with disrupted cost matrix must fail
        is_feasible = normal_sol.validate_feasibility(
            capacity=instance.vehicle_capacity,
            total_customers=instance.num_customers,
            cost_matrix=disrupted.cost_matrix
        )
        self.assertFalse(is_feasible)

if __name__ == '__main__':
    unittest.main()
