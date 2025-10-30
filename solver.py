# solver.py

import numpy as np
from scipy.optimize import linear_sum_assignment

# ... (solve_assignment_problem is unchanged)
def solve_assignment_problem(costs):
    cost_matrix = np.array(costs, dtype=float) # Ensure float type for inf
    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("Cost matrix for the assignment problem must be square.")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = list(zip(row_ind, col_ind))
    total_cost = cost_matrix[row_ind, col_ind].sum()
    solution_matrix = np.zeros_like(cost_matrix)
    solution_matrix[row_ind, col_ind] = 1
    steps = [{
        "description": "Assignment problem solved using the Hungarian method (via SciPy).\n\nThis optimized solver does not provide intermediate steps.",
        "matrix": solution_matrix,
        "highlight": []
    }]
    return assignments, total_cost, solution_matrix, steps


def north_west_corner(supply, demand):
    """
    Finds an initial solution using the North-West Corner Rule.
    NOTE: This method IGNORES costs, so it may allocate to a forbidden (infinite cost) cell.
    """
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)
    rows, cols = len(s), len(d)
    solution = np.zeros((rows, cols))
    steps = []
    i, j = 0, 0
    
    steps.append({
        "description": "Start at the North-West corner cell (1, 1).",
        "matrix": solution.copy(),
        "highlight": [(i, j)]
    })

    while i < rows and j < cols:
        desc = f"Considering cell ({i+1}, {j+1}).\n\n"
        desc += f"Supply at Source {i+1}: {s[i]}\n"
        desc += f"Demand at Destination {j+1}: {d[j]}"
        steps.append({ "description": desc, "matrix": solution.copy(), "highlight": [(i, j)] })

        shipment = min(s[i], d[j])
        solution[i, j] = shipment
        s[i] -= shipment
        d[j] -= shipment
        
        desc = f"Allocate min({s[i]+shipment}, {d[j]+shipment}) = {shipment} units."
        steps.append({ "description": desc, "matrix": solution.copy(), "highlight": [(i, j)] })

        if s[i] == 0 and d[j] == 0 and (i + 1 < rows or j + 1 < cols):
             desc = f"Supply for Source {i+1} is exhausted and Demand for Destination {j+1} is met.\nMoving diagonally to cell ({i+2}, {j+2})."
             i += 1; j += 1
        elif s[i] == 0:
            desc = f"Supply for Source {i+1} is exhausted.\nMoving to the next row, to cell ({i+2}, {j+1})."
            i += 1
        else:
            desc = f"Demand for Destination {j+1} is met.\nMoving to the next column, to cell ({i+1}, {j+2})."
            j += 1
        
        if i < rows and j < cols:
            steps.append({ "description": desc, "matrix": solution.copy(), "highlight": [(i, j)] })
            
    steps.append({ "description": "Process complete. Initial feasible solution found.", "matrix": solution.copy(), "highlight": []})
    return solution, steps

# ... (least_cost, vogel_approximation, and modi_method are functionally correct as NumPy
# handles np.inf appropriately in comparisons and sorting. Only need to ensure float dtypes.)
def least_cost(costs, supply, demand):
    c = np.array(costs, dtype=float)
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)
    rows, cols = c.shape
    solution = np.zeros((rows, cols))
    steps = []
    cost_tuples = sorted([(c[r, col], r, col) for r in range(rows) for col in range(cols)])
    steps.append({ "description": "Start with an empty allocation matrix.", "matrix": solution.copy(), "highlight": [] })
    for cost, i, j in cost_tuples:
        if s[i] > 0 and d[j] > 0:
            desc = f"Scanning for the lowest cost cell with available supply/demand.\n\nCell ({i+1}, {j+1}) has the lowest cost of {cost if cost != np.inf else '∞'}."
            steps.append({ "description": desc, "matrix": solution.copy(), "highlight": [(i, j)] })
            if cost == np.inf:
                steps.append({"description": "Skipping this cell as it represents a forbidden route.", "matrix": solution.copy(), "highlight": []})
                continue
            shipment = min(s[i], d[j])
            solution[i, j] = shipment
            desc = f"Supply: {s[i]}, Demand: {d[j]}.\nAllocating min({s[i]}, {d[j]}) = {shipment} units."
            s[i] -= shipment
            d[j] -= shipment
            steps.append({ "description": desc, "matrix": solution.copy(), "highlight": [(i, j)] })
    steps.append({ "description": "Process complete. Initial feasible solution found.", "matrix": solution.copy(), "highlight": []})
    return solution, steps

def vogel_approximation(costs, supply, demand):
    c = np.array(costs, dtype=float)
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)
    solution = np.zeros_like(c)
    steps = [{"description": "Start with an empty allocation matrix.", "matrix": solution.copy(), "highlight": []}]
    while np.sum(s) > 0 and np.sum(d) > 0:
        active_rows = np.where(s > 0)[0]
        active_cols = np.where(d > 0)[0]
        row_penalties, col_penalties = {}, {}
        for i in active_rows:
            row_costs = np.sort(c[i, active_cols])
            row_costs = row_costs[np.isfinite(row_costs)] # Exclude inf from penalty calculation
            row_penalties[i] = row_costs[1] - row_costs[0] if len(row_costs) > 1 else (row_costs[0] if len(row_costs) == 1 else -1)
        for j in active_cols:
            col_costs = np.sort(c[active_rows, j])
            col_costs = col_costs[np.isfinite(col_costs)]
            col_penalties[j] = col_costs[1] - col_costs[0] if len(col_costs) > 1 else (col_costs[0] if len(col_costs) == 1 else -1)
        rp_str = ", ".join([f"R{r+1}: {p:.1f}" for r, p in row_penalties.items()])
        cp_str = ", ".join([f"C{c+1}: {p:.1f}" for c, p in col_penalties.items()])
        desc = f"Calculate penalties.\n\nRow Penalties: {rp_str}\nCol Penalties: {cp_str}"
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": []})
        max_rp = max(row_penalties.values()) if row_penalties else -1
        max_cp = max(col_penalties.values()) if col_penalties else -1
        if max_rp >= max_cp:
            row_idx = max(row_penalties, key=row_penalties.get)
            min_cost_in_row = np.min(c[row_idx, active_cols])
            col_idx = active_cols[np.where(c[row_idx, active_cols] == min_cost_in_row)[0][0]]
            desc = f"Max penalty is {max_rp:.1f} in Row {row_idx+1}.\nLowest cost in this row is {min_cost_in_row if min_cost_in_row != np.inf else '∞'} at cell ({row_idx+1}, {col_idx+1})."
        else:
            col_idx = max(col_penalties, key=col_penalties.get)
            min_cost_in_col = np.min(c[active_rows, col_idx])
            row_idx = active_rows[np.where(c[active_rows, col_idx] == min_cost_in_col)[0][0]]
            desc = f"Max penalty is {max_cp:.1f} in Column {col_idx+1}.\nLowest cost is {min_cost_in_col if min_cost_in_col != np.inf else '∞'} at cell ({row_idx+1}, {col_idx+1})."
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": [(row_idx, col_idx)]})
        shipment = min(s[row_idx], d[col_idx])
        solution[row_idx, col_idx] = shipment
        desc = f"Supply: {s[row_idx]}, Demand: {d[col_idx]}.\nAllocating {shipment} units."
        s[row_idx] -= shipment; d[col_idx] -= shipment
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": [(row_idx, col_idx)]})
    steps.append({"description": "Process complete. Initial feasible solution found.", "matrix": solution.copy(), "highlight": []})
    return solution, steps

def modi_method(costs, initial_solution):
    c = np.array(costs, dtype=float)
    solution = np.array(initial_solution, dtype=float)
    # ... (modi_method and find_modi_loop from previous step are already compatible)
    steps = []
    iteration = 1
    while True:
        desc = f"--- MODI Iteration {iteration} ---\n\nStarting with the current allocation matrix."
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": []})
        basic_cells = np.argwhere(solution > 1e-9)
        u, v = np.full(c.shape[0], np.nan), np.full(c.shape[1], np.nan)
        u[0] = 0
        for _ in range(len(u) + len(v) + 1):
            for r, col in basic_cells:
                if not np.isnan(u[r]) and np.isnan(v[col]): v[col] = c[r, col] - u[r]
                elif np.isnan(u[r]) and not np.isnan(v[col]): u[r] = c[r, col] - v[col]
            if not np.any(np.isnan(u)) and not np.any(np.isnan(v)): break
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            steps.append({"description": "Degenerate solution detected. Cannot solve for all u and v. Stopping.", "matrix": solution.copy(), "highlight": []})
            break
        u_str = ", ".join([f'u{i+1}={val:.1f}' for i, val in enumerate(u)]); v_str = ", ".join([f'v{j+1}={val:.1f}' for j, val in enumerate(v)])
        desc = f"Calculated dual variables (u, v) from basic cells (where c[i,j] = u[i] + v[j]).\n\n{u_str}\n{v_str}"
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": []})
        opp_costs = np.zeros_like(c); is_optimal = True
        most_negative_cell, min_opp_cost = None, -1e-9
        non_basic_cells = np.argwhere(solution <= 1e-9)
        for r, col in non_basic_cells:
            cost = c[r, col] - (u[r] + v[col])
            opp_costs[r, col] = cost
            if cost < min_opp_cost: min_opp_cost, most_negative_cell = cost, (r, col); is_optimal = False
        desc = "Calculate opportunity costs (c[i,j] - u[i] - v[j]) for all non-basic (empty) cells."
        steps.append({"description": desc, "matrix": opp_costs, "highlight": [most_negative_cell] if most_negative_cell else []})
        if is_optimal:
            steps.append({"description": "All opportunity costs are non-negative. The solution is OPTIMAL.", "matrix": solution.copy(), "highlight": []})
            break
        desc = f"Most negative cost is {min_opp_cost:.2f} at cell {tuple(x+1 for x in most_negative_cell)}.\nThis is the entering cell."
        path = find_modi_loop(solution, most_negative_cell)
        if not path:
            steps.append({"description": "Could not find a closed loop. Stopping.", "matrix": solution.copy(), "highlight": [most_negative_cell]})
            break
        plus_cells, minus_cells = path[::2], path[1::2]
        desc = f"Found closed loop: {[tuple(x+1 for x in p) for p in path]}.\n\nGreen cells will increase, Red cells will decrease."
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": plus_cells, "highlight_minus": minus_cells})
        min_allocation = min(solution[r, col] for r, col in minus_cells)
        desc = f"The smallest allocation in the red (minus) cells is {min_allocation}.\n\nThis value will be added to green cells and subtracted from red cells."
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": plus_cells, "highlight_minus": minus_cells})
        for r, col in plus_cells: solution[r, col] += min_allocation
        for r, col in minus_cells: solution[r, col] -= min_allocation
        desc = "Allocations have been adjusted along the loop."
        steps.append({"description": desc, "matrix": solution.copy(), "highlight": []})
        iteration += 1
        if iteration > 20: steps.append({"description": "Reached max iterations.", "matrix": solution.copy(), "highlight": []}); break
    return solution, steps

def find_modi_loop(solution, start_node):
    # ... (unchanged)
    nodes = list(tuple(c) for c in np.argwhere(solution > 1e-9)) + [start_node]
    for r2, c2 in nodes:
        if r2 == start_node[0] or c2 == start_node[1]: continue
        if (start_node[0], c2) in nodes and (r2, start_node[1]) in nodes:
            return [start_node, (start_node[0], c2), (r2, c2), (r2, start_node[1])]
    return None
