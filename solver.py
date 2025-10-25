import numpy as np


def north_west_corner(supply, demand):
    """Finds an initial solution using the North-West Corner Rule."""
    supply = np.array(supply)
    demand = np.array(demand)
    rows, cols = len(supply), len(demand)
    solution = np.zeros((rows, cols))
    i, j = 0, 0
    while i < rows and j < cols:
        shipment = min(supply[i], demand[j])
        solution[i, j] = shipment
        supply[i] -= shipment
        demand[j] -= shipment
        if supply[i] == 0:
            i += 1
        else:
            j += 1
    return solution


def least_cost(costs, supply, demand):
    """Finds an initial solution using the Least Cost Method."""
    costs = np.array(costs)
    supply = np.array(supply)
    demand = np.array(demand)
    rows, cols = costs.shape
    solution = np.zeros((rows, cols))

    cost_indices = np.argsort(costs.flatten())

    for index in cost_indices:
        i, j = np.unravel_index(index, costs.shape)
        if supply[i] > 0 and demand[j] > 0:
            shipment = min(supply[i], demand[j])
            solution[i, j] = shipment
            supply[i] -= shipment
            demand[j] -= shipment

    return solution


def vogel_approximation(costs, supply, demand):
    """Finds an initial solution using Vogel's Approximation Method (VAM)."""
    costs = np.array(costs)
    supply = np.array(supply)
    demand = np.array(demand)
    rows, cols = costs.shape
    solution = np.zeros((rows, cols))

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        row_penalty = []
        for i in range(rows):
            if supply[i] > 0:
                sorted_costs = sorted([costs[i, j] for j in range(cols) if demand[j] > 0])
                row_penalty.append(sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else sorted_costs[0])
            else:
                row_penalty.append(-np.inf)

        col_penalty = []
        for j in range(cols):
            if demand[j] > 0:
                sorted_costs = sorted([costs[i, j] for i in range(rows) if supply[i] > 0])
                col_penalty.append(sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else sorted_costs[0])
            else:
                col_penalty.append(-np.inf)

        max_row_penalty = max(row_penalty)
        max_col_penalty = max(col_penalty)

        if max_row_penalty >= max_col_penalty:
            row_idx = np.argmax(row_penalty)
            min_cost_col = -1
            min_cost = np.inf
            for j in range(cols):
                if demand[j] > 0 and costs[row_idx, j] < min_cost:
                    min_cost = costs[row_idx, j]
                    min_cost_col = j

            i, j = row_idx, min_cost_col
        else:
            col_idx = np.argmax(col_penalty)
            min_cost_row = -1
            min_cost = np.inf
            for i in range(rows):
                if supply[i] > 0 and costs[i, col_idx] < min_cost:
                    min_cost = costs[i, col_idx]
                    min_cost_row = i
            i, j = min_cost_row, col_idx

        shipment = min(supply[i], demand[j])
        solution[i, j] = shipment
        supply[i] -= shipment
        demand[j] -= shipment

    return solution


def modi_method(costs, initial_solution):
    """Optimizes a solution using the Modified Distribution (MODI) method."""
    costs = np.array(costs)
    solution = np.array(initial_solution)

    while True:
        # 1. Find basic (allocated) and non-basic cells
        basic_cells = np.argwhere(solution > 0)

        # 2. Calculate u and v values
        u = np.full(costs.shape[0], np.nan)
        v = np.full(costs.shape[1], np.nan)
        u[0] = 0

        # Iteratively find all u and v values
        while np.any(np.isnan(u)) or np.any(np.isnan(v)):
            for r, c in basic_cells:
                if not np.isnan(u[r]) and np.isnan(v[c]):
                    v[c] = costs[r, c] - u[r]
                elif np.isnan(u[r]) and not np.isnan(v[c]):
                    u[r] = costs[r, c] - v[c]

        # 3. Calculate opportunity costs for non-basic cells
        opportunity_costs = np.zeros_like(costs, dtype=float)
        is_optimal = True
        most_negative_cell = None
        min_opp_cost = 0

        for r in range(costs.shape[0]):
            for c in range(costs.shape[1]):
                if solution[r, c] == 0:
                    opp_cost = costs[r, c] - (u[r] + v[c])
                    opportunity_costs[r, c] = opp_cost
                    if opp_cost < min_opp_cost:
                        min_opp_cost = opp_cost
                        most_negative_cell = (r, c)
                        is_optimal = False

        # 4. If all opportunity costs are non-negative, solution is optimal
        if is_optimal:
            break

        # 5. Find a closed loop to improve the solution
        start_node = most_negative_cell
        path = find_modi_loop(solution, start_node)

        # 6. Adjust allocations along the loop
        plus_cells = path[::2]
        minus_cells = path[1::2]

        min_allocation = min(solution[r, c] for r, c in minus_cells)

        for r, c in plus_cells:
            solution[r, c] += min_allocation
        for r, c in minus_cells:
            solution[r, c] -= min_allocation

    return solution


def find_modi_loop(solution, start_node):
    """Helper function for MODI to find a closed path (loop)."""
    # This is a graph traversal problem. A simple implementation for common cases:
    # Build a graph of basic cells. An edge exists between cells in the same row or col.
    nodes = list(tuple(c) for c in np.argwhere(solution > 0)) + [start_node]

    # Simple path finding for now. For complex cases, a more robust graph algorithm is needed.
    # We are looking for a path that alternates between horizontal and vertical moves.

    # A simplified search that finds rectangular loops
    for r2, c2 in nodes:
        if r2 == start_node[0] or c2 == start_node[1]: continue
        # Check if a rectangle can be formed: (r1, c1) -> (r1, c2) -> (r2, c2) -> (r2, c1)
        if (start_node[0], c2) in nodes and (r2, start_node[1]) in nodes:
            return [start_node, (start_node[0], c2), (r2, c2), (r2, start_node[1])]

    # A more general but complex search would be required for non-rectangular loops.
    # For this application, we assume non-degenerate solutions where simple loops exist.
    raise Exception("Could not find a closed loop. The problem might be degenerate.")