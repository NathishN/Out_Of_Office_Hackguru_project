import time
import copy
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from src.routing_model import ProblemInstance, Solution, Route
from src.baseline_optimizer import apply_two_opt

class SimulatedAnnealingOptimizer:
    """
    Quantum-inspired Simulated Annealing optimizer for CVRP.
    Uses soft penalty terms for overcapacity and unassigned deliveries.
    """
    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
        max_iterations: int = 5000,
        penalty_capacity: float = 100.0,
        penalty_unassigned: float = 500.0,
        seed: int = 42
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.penalty_capacity = penalty_capacity
        self.penalty_unassigned = penalty_unassigned
        self.seed = seed

    def evaluate_solution(self, routes_node_ids: List[List[int]], instance: ProblemInstance) -> Tuple[float, float, int, int]:
        """
        Evaluates solution objective value including soft penalties.
        Returns: (total_energy, route_cost, capacity_violations, unassigned_count)
        """
        cost_matrix = instance.cost_matrix
        total_route_cost = 0.0
        capacity_penalty = 0.0
        capacity_violations = 0
        visited_customers = set()

        for route in routes_node_ids:
            # Route cost
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                total_route_cost += cost_matrix[u, v]
            
            # Route capacity check
            route_demand = sum(instance.get_node_by_id(nid).demand for nid in route if nid != 0)
            if route_demand > instance.vehicle_capacity:
                excess = route_demand - instance.vehicle_capacity
                capacity_penalty += excess * self.penalty_capacity
                capacity_violations += 1

            for nid in route:
                if nid != 0:
                    visited_customers.add(nid)

        unassigned_count = instance.num_customers - len(visited_customers)
        unassigned_penalty = unassigned_count * self.penalty_unassigned

        total_energy = total_route_cost + capacity_penalty + unassigned_penalty
        return total_energy, total_route_cost, capacity_violations, unassigned_count

    def generate_neighbor(self, current_routes: List[List[int]], num_vehicles: int, num_customers: int) -> List[List[int]]:
        """
        Generates a neighbor solution using one of three move operators:
        1. Relocate a customer to another route.
        2. Swap two customers between different routes.
        3. Perform intra-route 2-opt.
        """
        neighbor = [list(r) for r in current_routes]
        move_type = random.choice(["relocate", "swap", "2opt"])

        non_empty_routes = [idx for idx, r in enumerate(neighbor) if len(r) > 2]
        if not non_empty_routes:
            return neighbor

        if move_type == "relocate":
            r_src_idx = random.choice(non_empty_routes)
            r_src = neighbor[r_src_idx]
            customer_pos = random.randint(1, len(r_src) - 2)
            customer = r_src.pop(customer_pos)

            r_dst_idx = random.randint(0, num_vehicles - 1)
            r_dst = neighbor[r_dst_idx]
            insert_pos = random.randint(1, len(r_dst) - 1)
            r_dst.insert(insert_pos, customer)

        elif move_type == "swap":
            if len(non_empty_routes) >= 2:
                r1_idx, r2_idx = random.sample(non_empty_routes, 2)
                r1, r2 = neighbor[r1_idx], neighbor[r2_idx]
                if len(r1) > 2 and len(r2) > 2:
                    pos1 = random.randint(1, len(r1) - 2)
                    pos2 = random.randint(1, len(r2) - 2)
                    r1[pos1], r2[pos2] = r2[pos2], r1[pos1]
            elif len(non_empty_routes) == 1:
                r1 = neighbor[non_empty_routes[0]]
                if len(r1) > 3:
                    pos1, pos2 = random.sample(range(1, len(r1) - 1), 2)
                    r1[pos1], r1[pos2] = r1[pos2], r1[pos1]

        elif move_type == "2opt":
            r_idx = random.choice(non_empty_routes)
            r = neighbor[r_idx]
            if len(r) > 4:
                i = random.randint(1, len(r) - 3)
                k = random.randint(i + 1, len(r) - 2)
                neighbor[r_idx] = r[:i] + r[i:k + 1][::-1] + r[k + 1:]

        return neighbor

    def solve(self, instance: ProblemInstance) -> Solution:
        start_time = time.perf_counter()
        random.seed(self.seed)

        # Initial solution: Distribute customers across vehicles
        num_vehicles = instance.num_vehicles
        all_customers = [c.id for c in instance.customers]
        random.shuffle(all_customers)

        current_routes = [[0, 0] for _ in range(num_vehicles)]
        for idx, customer_id in enumerate(all_customers):
            vehicle_idx = idx % num_vehicles
            current_routes[vehicle_idx].insert(-1, customer_id)

        best_routes = copy.deepcopy(current_routes)
        best_energy, best_cost, _, _ = self.evaluate_solution(best_routes, instance)
        current_energy = best_energy

        temp = self.initial_temp
        history: List[float] = []

        for iteration in range(self.max_iterations):
            if temp <= self.min_temp:
                break

            candidate_routes = self.generate_neighbor(current_routes, num_vehicles, instance.num_customers)
            cand_energy, cand_cost, cand_violations, cand_unassigned = self.evaluate_solution(candidate_routes, instance)

            delta_e = cand_energy - current_energy

            # Metropolis Acceptance Rule
            if delta_e <= 0 or random.random() < np.exp(-delta_e / temp):
                current_routes = candidate_routes
                current_energy = cand_energy

                if cand_energy < best_energy:
                    best_routes = copy.deepcopy(candidate_routes)
                    best_energy = cand_energy

            temp *= self.cooling_rate
            history.append(current_energy)

        # Apply 2-opt post-processing to best routes
        final_routes: List[Route] = []
        visited = set()
        for v_id, r_nodes in enumerate(best_routes):
            r_cleaned = apply_two_opt(r_nodes, instance.cost_matrix)
            demand = sum(instance.get_node_by_id(nid).demand for nid in r_cleaned if nid != 0)
            route_obj = Route(vehicle_id=v_id, node_ids=r_cleaned, total_demand=demand)
            route_obj.compute_route_cost(instance.cost_matrix)
            final_routes.append(route_obj)
            for nid in r_cleaned:
                if nid != 0:
                    visited.add(nid)

        unassigned = [c.id for c in instance.customers if c.id not in visited]
        end_time = time.perf_counter()

        solution = Solution(
            routes=final_routes,
            unassigned_nodes=unassigned,
            execution_time_ms=(end_time - start_time) * 1000.0,
            algorithm_name="Simulated Annealing (Quantum-Inspired)"
        )
        solution.compute_total_cost(instance.cost_matrix)
        solution.validate_feasibility(instance.vehicle_capacity, instance.num_customers, cost_matrix=instance.cost_matrix)
        return solution
