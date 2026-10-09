import time
import numpy as np
from typing import List, Tuple
from src.routing_model import ProblemInstance, Solution, Route

def two_opt_swap(route_nodes: List[int], i: int, k: int) -> List[int]:
    """
    Performs 2-opt swap reversing route subsegment between indices i and k.
    route_nodes includes depot at start and end.
    """
    new_route = route_nodes[:i] + route_nodes[i:k + 1][::-1] + route_nodes[k + 1:]
    return new_route

def apply_two_opt(route_nodes: List[int], cost_matrix: np.ndarray) -> List[int]:
    """
    Iteratively improves a single route using 2-opt local search.
    """
    if len(route_nodes) <= 3:
        return route_nodes

    best_route = list(route_nodes)
    improved = True

    while improved:
        improved = False
        n = len(best_route)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                # Calculate cost change
                u1, v1 = best_route[i - 1], best_route[i]
                u2, v2 = best_route[k], best_route[k + 1]

                old_dist = cost_matrix[u1, v1] + cost_matrix[u2, v2]
                new_dist = cost_matrix[u1, u2] + cost_matrix[v1, v2]

                if new_dist < old_dist - 1e-6 and cost_matrix[u1, u2] < 1e5 and cost_matrix[v1, v2] < 1e5:
                    best_route = two_opt_swap(best_route, i, k)
                    improved = True
                    break
            if improved:
                break

    return best_route

class BaselineOptimizer:
    """
    Traditional Nearest Neighbor heuristic with optional 2-opt local search enhancement.
    """
    def __init__(self, use_two_opt: bool = True):
        self.use_two_opt = use_two_opt

    def solve(self, instance: ProblemInstance) -> Solution:
        start_time = time.perf_counter()
        
        cost_matrix = instance.cost_matrix
        unvisited = set(c.id for c in instance.customers)
        routes: List[Route] = []

        for v_id in range(instance.num_vehicles):
            if not unvisited:
                break

            current_node = 0  # Start at depot
            current_capacity = instance.vehicle_capacity
            route_node_ids = [0]
            route_demand = 0

            while unvisited:
                # Find nearest feasible unvisited customer
                best_next = None
                best_dist = float('inf')

                for candidate_id in unvisited:
                    demand = instance.get_node_by_id(candidate_id).demand
                    if demand <= current_capacity:
                        dist = cost_matrix[current_node, candidate_id]
                        if dist < best_dist and dist < 1e5:
                            best_dist = dist
                            best_next = candidate_id

                if best_next is None:
                    # No customer fits remaining vehicle capacity
                    break

                # Assign customer to current vehicle route
                unvisited.remove(best_next)
                route_node_ids.append(best_next)
                demand = instance.get_node_by_id(best_next).demand
                current_capacity -= demand
                route_demand += demand
                current_node = best_next

            # Return to depot
            route_node_ids.append(0)

            # Apply 2-opt local search improvement
            if self.use_two_opt and len(route_node_ids) > 3:
                route_node_ids = apply_two_opt(route_node_ids, cost_matrix)

            route = Route(vehicle_id=v_id, node_ids=route_node_ids, total_demand=route_demand)
            route.compute_route_cost(cost_matrix)
            routes.append(route)

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000.0

        unassigned = list(unvisited)
        solution = Solution(
            routes=routes,
            unassigned_nodes=unassigned,
            execution_time_ms=execution_time_ms,
            algorithm_name="Nearest Neighbor + 2-opt" if self.use_two_opt else "Nearest Neighbor"
        )
        solution.compute_total_cost(cost_matrix)
        solution.validate_feasibility(instance.vehicle_capacity, instance.num_customers, cost_matrix=cost_matrix)
        return solution
