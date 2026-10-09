import unittest
import math
import numpy as np
from src.data_generator import generate_cvrp_instance

class TestDataGenerator(unittest.TestCase):
    def test_generate_instance_properties(self):
        instance = generate_cvrp_instance(num_customers=10, num_vehicles=2, vehicle_capacity=40, seed=42)
        self.assertEqual(instance.num_customers, 10)
        self.assertEqual(instance.total_nodes, 11)
        self.assertEqual(instance.depot.id, 0)
        self.assertTrue(instance.depot.is_depot)
        self.assertEqual(len(instance.customers), 10)
        self.assertEqual(instance.cost_matrix.shape, (11, 11))

    def test_reproducibility(self):
        inst1 = generate_cvrp_instance(num_customers=5, seed=123)
        inst2 = generate_cvrp_instance(num_customers=5, seed=123)
        np.testing.assert_array_almost_equal(inst1.cost_matrix, inst2.cost_matrix)
        for c1, c2 in zip(inst1.customers, inst2.customers):
            self.assertEqual(c1.demand, c2.demand)
            self.assertAlmostEqual(c1.x, c2.x)
            self.assertAlmostEqual(c1.y, c2.y)

    def test_few_delivery_locations(self):
        instance = generate_cvrp_instance(num_customers=3, num_vehicles=2, vehicle_capacity=50, seed=99)
        self.assertEqual(instance.num_customers, 3)
        self.assertEqual(instance.total_nodes, 4)
        self.assertEqual(instance.cost_matrix.shape, (4, 4))

    def test_demands_and_coordinates_bounds(self):
        instance = generate_cvrp_instance(
            num_customers=15,
            grid_size=100.0,
            min_demand=5,
            max_demand=20,
            seed=42
        )
        for c in instance.customers:
            self.assertGreaterEqual(c.demand, 5)
            self.assertLessEqual(c.demand, 20)
            self.assertGreaterEqual(c.x, 0.0)
            self.assertLessEqual(c.x, 100.0)
            self.assertGreaterEqual(c.y, 0.0)
            self.assertLessEqual(c.y, 100.0)

    def test_distance_calculation(self):
        instance = generate_cvrp_instance(num_customers=5, seed=42)
        n0 = instance.get_node_by_id(0)
        n1 = instance.get_node_by_id(1)
        expected_dist = math.hypot(n0.x - n1.x, n0.y - n1.y)
        self.assertAlmostEqual(instance.cost_matrix[0, 1], expected_dist, places=5)
        self.assertAlmostEqual(instance.cost_matrix[1, 0], expected_dist, places=5)

if __name__ == '__main__':
    unittest.main()
