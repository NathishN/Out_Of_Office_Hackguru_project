import numpy as np
from typing import Optional
from src.routing_model import Node, ProblemInstance

def generate_cvrp_instance(
    num_customers: int = 15,
    num_vehicles: int = 3,
    vehicle_capacity: int = 50,
    grid_size: float = 100.0,
    min_demand: int = 5,
    max_demand: int = 20,
    seed: Optional[int] = 42
) -> ProblemInstance:
    """
    Generates a reproducible synthetic CVRP problem instance.
    Depot is positioned at the center (grid_size / 2, grid_size / 2).
    """
    if seed is not None:
        np.random.seed(seed)

    depot_x, depot_y = grid_size / 2.0, grid_size / 2.0
    depot = Node(id=0, x=depot_x, y=depot_y, demand=0, is_depot=True)

    customers = []
    for i in range(1, num_customers + 1):
        x = float(np.random.uniform(0, grid_size))
        y = float(np.random.uniform(0, grid_size))
        demand = int(np.random.randint(min_demand, max_demand + 1))
        customers.append(Node(id=i, x=x, y=y, demand=demand, is_depot=False))

    all_nodes = [depot] + customers
    n = len(all_nodes)
    cost_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i, j] = all_nodes[i].distance_to(all_nodes[j])
            else:
                cost_matrix[i, j] = 0.0

    return ProblemInstance(
        depot=depot,
        customers=customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        cost_matrix=cost_matrix
    )
