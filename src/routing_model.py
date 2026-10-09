from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class Node:
    """
    Represents a location in the delivery network (depot or customer).
    """
    id: int
    x: float
    y: float
    demand: int = 0
    is_depot: bool = False

    def distance_to(self, other: "Node") -> float:
        """Calculates Euclidean distance to another node."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

@dataclass
class Route:
    """
    Represents an ordered route assigned to a single vehicle.
    node_ids should start and end with the depot ID (0).
    """
    vehicle_id: int
    node_ids: List[int] = field(default_factory=list)
    total_demand: int = 0
    total_cost: float = 0.0

    def compute_route_cost(self, cost_matrix: np.ndarray) -> float:
        """Computes total cost of the route given a cost matrix."""
        if len(self.node_ids) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(self.node_ids) - 1):
            u, v = self.node_ids[i], self.node_ids[i + 1]
            cost += cost_matrix[u, v]
        self.total_cost = float(cost)
        return self.total_cost

@dataclass
class Solution:
    """
    Represents a complete CVRP solution across the fleet.
    """
    routes: List[Route] = field(default_factory=list)
    total_cost: float = 0.0
    is_feasible: bool = True
    unassigned_nodes: List[int] = field(default_factory=list)
    execution_time_ms: float = 0.0
    algorithm_name: str = "Unknown"

    def compute_total_cost(self, cost_matrix: np.ndarray) -> float:
        """Calculates and updates total cost across all vehicle routes."""
        self.total_cost = sum(r.compute_route_cost(cost_matrix) for r in self.routes)
        return self.total_cost

    def validate_feasibility(
        self,
        capacity: int,
        total_customers: int,
        cost_matrix: "np.ndarray | None" = None
    ) -> bool:
        """
        Validates if the solution satisfies capacity constraints, customer coverage, no unassigned nodes,
        and no traversal of unavailable/blocked edges.
        """
        self.is_feasible = True
        
        # Check unassigned nodes
        if self.unassigned_nodes:
            self.is_feasible = False
            return False

        # Check vehicle capacity constraints
        for route in self.routes:
            if route.total_demand > capacity:
                self.is_feasible = False
                return False

        # Check customer coverage
        visited = []
        for route in self.routes:
            # exclude depot nodes (start and end)
            customers_in_route = [nid for nid in route.node_ids if nid != 0]
            visited.extend(customers_in_route)

        if len(visited) != total_customers or len(set(visited)) != total_customers:
            self.is_feasible = False
            return False

        # Check unavailable / blocked edges (cost >= 1e5)
        if cost_matrix is not None:
            for route in self.routes:
                for i in range(len(route.node_ids) - 1):
                    u, v = route.node_ids[i], route.node_ids[i + 1]
                    if cost_matrix[u, v] >= 1e5:
                        self.is_feasible = False
                        return False

        return self.is_feasible

@dataclass
class ProblemInstance:
    """
    Encapsulates all data required for a CVRP problem instance.
    """
    depot: Node
    customers: List[Node]
    num_vehicles: int
    vehicle_capacity: int
    cost_matrix: np.ndarray
    disrupted_edges: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def num_customers(self) -> int:
        return len(self.customers)

    @property
    def total_nodes(self) -> int:
        return len(self.customers) + 1

    def get_node_by_id(self, node_id: int) -> Node:
        if node_id == 0:
            return self.depot
        for customer in self.customers:
            if customer.id == node_id:
                return customer
        raise ValueError(f"Node with ID {node_id} not found.")

    def get_all_nodes(self) -> List[Node]:
        return [self.depot] + self.customers
