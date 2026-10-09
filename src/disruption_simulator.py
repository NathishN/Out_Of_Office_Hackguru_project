import numpy as np
import copy
from typing import Tuple, List, Dict, Any
from src.routing_model import ProblemInstance, Solution

BLOCKED_COST = 1e6  # Large finite representation for unavailable edge

class DisruptionSimulator:
    """
    Simulates operational disruptions on CVRP instances and evaluates route impact.
    """

    @staticmethod
    def apply_edge_closure(instance: ProblemInstance, u: int, v: int) -> ProblemInstance:
        """
        Marks road segment (u, v) and (v, u) as blocked (cost = 1,000,000).
        """
        new_instance = copy.deepcopy(instance)
        new_instance.cost_matrix[u, v] = BLOCKED_COST
        new_instance.cost_matrix[v, u] = BLOCKED_COST
        new_instance.disrupted_edges.append((u, v))
        return new_instance

    @staticmethod
    def apply_traffic_penalty(instance: ProblemInstance, u: int, v: int, multiplier: float = 5.0) -> ProblemInstance:
        """
        Increases travel cost on road segment (u, v) and (v, u) by a penalty factor.
        """
        new_instance = copy.deepcopy(instance)
        new_instance.cost_matrix[u, v] *= multiplier
        new_instance.cost_matrix[v, u] *= multiplier
        new_instance.disrupted_edges.append((u, v))
        return new_instance

    @staticmethod
    def apply_vehicle_breakdown(instance: ProblemInstance, breakdown_vehicle_id: int) -> ProblemInstance:
        """
        Simulates a breakdown by reducing available fleet size by 1.
        """
        new_instance = copy.deepcopy(instance)
        new_instance.num_vehicles = max(1, new_instance.num_vehicles - 1)
        return new_instance

    @staticmethod
    def check_affected_routes(solution: Solution, disrupted_edges: List[Tuple[int, int]]) -> List[int]:
        """
        Returns indices of routes in the solution that traverse any disrupted edges.
        """
        affected_indices = []
        for r_idx, route in enumerate(solution.routes):
            nodes = route.node_ids
            has_affected = False
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if (u, v) in disrupted_edges or (v, u) in disrupted_edges:
                    has_affected = True
                    break
            if has_affected:
                affected_indices.append(r_idx)
        return affected_indices
