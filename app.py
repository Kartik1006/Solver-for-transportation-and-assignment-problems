# transportation_app.py
"""
PySimpleGUI front-end for the enhanced Transportation Problem solver.

Features added:
- Vogel's Approximation Method (VAM) as an initial method option
- MODI optimization (automatic iterations until optimality)
- Interactive allocation grid: after an initial solution is computed, allocations are shown in an editable grid;
  the user can change allocations (positive numeric) and click "Apply Edits" to re-validate and recompute cost.
- "Optimize (MODI)" button runs the optimization and shows iteration logs.
- Input validation, friendly error popups, heatmap visualization (matplotlib), export to CSV/Excel.
"""

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
from typing import List

from solver import (
    validate_inputs, balance_problem, is_balanced,
    north_west_corner, least_cost_method, vogels_approximation_method,
    solution_to_dataframe, cost_of_solution, modi_iteration, optimize_by_modi
)


# ---------- Helper to draw matplotlib figure onto PySimpleGUI Canvas ----------
def draw_figure(canvas, figure):
    """Draw a matplotlib figure onto a Tk Canvas (used by PySimpleGUI)."""
    # Clear previous children
    for child in canvas.winfo_children():
        child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    widget.pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


# ---------- Grid builder & parser ----------
def make_input_grid(num_sources, num_destinations, default_cost=0):
    """Construct cost grid inputs + supply/demand inputs"""
    heading = [[sg.Text("Costs matrix (rows = sources, columns = destinations)", font=("Segoe UI", 10, "bold"))]]
    header_row = [sg.Text("Sources \\ Dest", size=(12,1))]
    for j in range(num_destinations):
        header_row.append(sg.Text(f"D{j+1}", size=(8,1)))
    header_row.append(sg.Text("Supply", size=(8,1)))
    rows = [heading, header_row]

    for i in range(num_sources):
        row = [sg.Text(f"S{i+1}", size=(12,1))]
        for j in range(num_destinations):
            key = f"cost_{i}_{j}"
            row.append(sg.Input(default_text=str(default_cost), size=(8,1), key=key))
        row.append(sg.Input(default_text="0", size=(8,1), key=f"supply_{i}"))
        rows.append(row)

    demand_row = [sg.Text("Demand", size=(12,1))]
    for j in range(num_destinations):
        demand_row.append(sg.Input(default_text="0", size=(8,1), key=f"demand_{j}"))
    demand_row.append(sg.Text("", size=(8,1)))
    rows.append(demand_row)

    return rows


def parse_grid_values(values, num_sources, num_destinations):
    """Parse the manual grid inputs to structured data"""
    try:
        sources = [f"S{i+1}" for i in range(num_sources)]
        destinations = [f"D{j+1}" for j in range(num_destinations)]
        costs = []
        supply = []
        demand = []

        for i in range(num_sources):
            row = []
            for j in range(num_destinations):
                key = f"cost_{i}_{j}"
                v = values.get(key)
                if v is None or v == "":
                    raise ValueError(f"Cost at row {i+1}, col {j+1} cannot be empty.")
                row.append(float(v))
            costs.append(row)
            sup_val = values.get(f"supply_{i}")
            if sup_val is None or sup_val == "":
                raise ValueError(f"Supply for source S{i+1} cannot be empty.")
            supply.append(float(sup_val))

        for j in range(num_destinations):
            dem_val = values.get(f"demand_{j}")
            if dem_val is None or dem_val == "":
                raise ValueError(f"Demand for destination D{j+1} cannot be empty.")
            demand.append(float(dem_val))

        return sources, destinations, costs, supply, demand
    except ValueError as e:
        raise e
    except Exception:
        raise ValueError("Please ensure all inputs are numeric and non-empty.")


# ---------- File import (same heuristic as before) ----------
def read_input_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path, header=None)
        elif ext == ".csv":
            df = pd.read_csv(path, header=None)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel (.xls/.xlsx).")
    except Exception:
        raise ValueError("Unable to read file. Please check the file and try again.")

    df_trim = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    data = df_trim.values
    rows, cols = data.shape
    numeric = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            try:
                float(str(data[i, j]))
                numeric[i, j] = True
            except Exception:
                numeric[i, j] = False

    if numeric.all():
        raise ValueError("The uploaded file appears to contain only numeric values. Please provide supply/demand or use manual input.")

    try:
        last_row = df_trim.iloc[-1, :].astype(str).str.lower().str.strip().tolist()
        demand_idx = None
        for idx, val in enumerate(last_row):
            if "demand" in val:
                demand_idx = idx
                break
        if demand_idx is not None or "demand" in "".join(last_row):
            numeric_block = df_trim.iloc[0:-1, 1:-1]
            costs = numeric_block.astype(float).values.tolist()
            supply = df_trim.iloc[0:-1, -1].astype(float).tolist()
            demand = df_trim.iloc[-1, 1:-1].astype(float).tolist()
            sources = [str(x) if x is not None else f"S{i+1}" for i, x in enumerate(df_trim.iloc[0:-1, 0].tolist())]
            destinations = [f"D{j+1}" for j in range(len(costs[0]))]
            return sources, destinations, costs, supply, demand
    except Exception:
        pass

    raise ValueError("Could not parse the uploaded file. Use the manual grid or upload a file with supply as last column and demand as last row.")


# ---------- GUI main ----------
def main():
    sg.theme("SystemDefault")

    # Top controls
    control_col = [
        [sg.Text("Transportation Solver — Enhanced", font=("Segoe UI", 14, "bold"))],
        [sg.Text("Sources:"), sg.Input("3", size=(5,1), key="-NS-"),
         sg.Text("Destinations:"), sg.Input("3", size=(5,1), key="-ND-"),
         sg.Button("Create Grid", key="-CREATE-")],
        [sg.Button("Load CSV/Excel", key="-LOAD-"), sg.Input(key="-FILEPATH-", visible=False)],
        [sg.Text("Initial Method:"), sg.Combo(["North-West Corner", "Least Cost", "Vogel (VAM)"], default_value="Vogel (VAM)", key="-METHOD-")],
        [sg.Button("Solve", key="-SOLVE-"), sg.Button("Optimize (MODI)", key="-MODI-", disabled=True)],
        [sg.Button("Apply Manual Edits", key="-APPLY-", disabled=True), sg.Button("Export Result", key="-EXPORT-", disabled=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("", key="-STATUS-", size=(80,1))]
    ]

    grid_col = sg.Column([[]], key="-GRIDCOL-", scrollable=True, vertical_scroll_only=True, size=(700, 320))
    plot_col = sg.Column([[]], key="-PLOTCOL-")

    layout = [
        [sg.Column(control_col), sg.VerticalSeparator(), sg.Column([[grid_col], [plot_col]])],
    ]

    window = sg.Window("Transportation Solver — Enhanced", layout, finalize=True, resizable=True)

    current_num_sources = 3
    current_num_destinations = 3
    window["-GRIDCOL-"].update(make_input_grid(current_num_sources, current_num_destinations))

    # State holders
    current_sources = None
    current_destinations = None
    current_costs = None
    current_supply = None
    current_demand = None
    current_solution = None
    current_total_cost = None

    fig_agg = None

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break

        if event == "-CREATE-":
            try:
                ns = int(values["-NS-"])
                nd = int(values["-ND-"])
                if ns <= 0 or nd <= 0:
                    sg.popup_error("Number of sources and destinations must be positive.")
                    continue
                current_num_sources = ns
                current_num_destinations = nd
                window["-GRIDCOL-"].update(make_input_grid(ns, nd, default_cost=0))
                window["-STATUS-"].update(f"Created grid: {ns} x {nd}. Enter values.")
            except Exception:
                sg.popup_error("Invalid numbers. Please enter integers for sources/destinations.")

        if event == "-LOAD-":
            path = sg.popup_get_file("Select CSV or Excel file", file_types=(("CSV Files", "*.csv"), ("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")))
            if path:
                try:
                    srcs, dests, costs, supply, demand = read_input_file(path)
                    ns = len(srcs)
                    nd = len(dests)
                    current_num_sources = ns
                    current_num_destinations = nd
                    window["-GRIDCOL-"].update(make_input_grid(ns, nd, default_cost=0))
                    for i in range(ns):
                        for j in range(nd):
                            window[f"cost_{i}_{j}"].update(str(costs[i][j]))
                        window[f"supply_{i}"].update(str(supply[i]))
                    for j in range(nd):
                        window[f"demand_{j}"].update(str(demand[j]))
                    window["-STATUS-"].update(f"Loaded: {os.path.basename(path)}")
                except Exception as e:
                    sg.popup_error("File load error", str(e))

        if event == "-SOLVE-":
            try:
                sources, destinations, costs, supply, demand = parse_grid_values(values, current_num_sources, current_num_destinations)
            except Exception as e:
                sg.popup_error("Input error", str(e))
                continue

            valid, msg = validate_inputs(sources, destinations, costs, supply, demand)
            if not valid:
                sg.popup_error("Validation error", msg)
                continue

            if not is_balanced(supply, demand):
                ans = sg.popup_yes_no("Problem unbalanced (total supply != total demand). Auto-balance with dummy row/col?")
                if ans != "Yes":
                    continue

            try:
                s2, d2, c2, sup2, dem2, was_balanced = balance_problem(sources, destinations, costs, supply, demand)
            except Exception as e:
                sg.popup_error("Balancing error", str(e))
                continue

            method = values["-METHOD-"]
            try:
                if method == "North-West Corner":
                    sol, total, logs = north_west_corner(s2, d2, c2, sup2, dem2)
                elif method == "Least Cost":
                    sol, total, logs = least_cost_method(s2, d2, c2, sup2, dem2)
                else:
                    sol, total, logs = vogels_approximation_method(s2, d2, c2, sup2, dem2)
            except Exception:
                sg.popup_error("Solver error", "Unexpected error while solving. Check inputs.")
                continue

            # Save state
            current_sources = s2
            current_destinations = d2
            current_costs = c2
            current_supply = sup2
            current_demand = dem2
            current_solution = sol
            current_total_cost = total

            # Show a popup summary (allocations + total)
            summary_lines = [f"Method: {method}", f"Total Cost: {total:.4f}", "Allocations:"]
            # produce allocation logs from solution:
            for i in range(len(s2)):
                for j in range(len(d2)):
                    if current_solution[i][j] != 0:
                        summary_lines.append(f"{s2[i]} -> {d2[j]} : {current_solution[i][j]} (c={c2[i][j]})")
            sg.popup_scrolled("\n".join(summary_lines), title="Initial Solution", size=(70,20))

            # Show interactive allocation grid (inputs editable)
            alloc_grid = [[sg.Text("Allocations (editable)", font=("Segoe UI", 10, "bold"))]]
            head = [sg.Text("Sources \\ Dest", size=(12,1))]
            for j in range(len(d2)):
                head.append(sg.Text(d2[j], size=(8,1)))
            head.append(sg.Text("Remaining Supply", size=(12,1)))
            alloc_grid.append(head)
            # compute remaining supply for display
            for i in range(len(s2)):
                row = [sg.Text(s2[i], size=(12,1))]
                for j in range(len(d2)):
                    val = current_solution[i][j]
                    row.append(sg.Input(default_text=str(val), size=(8,1), key=f"alloc_{i}_{j}"))
                # remaining supply
                remaining = current_supply[i] - sum(current_solution[i])
                row.append(sg.Text(f"{remaining:.2f}", size=(12,1), key=f"rem_sup_{i}"))
                alloc_grid.append(row)
            # demand row
            dem_row = [sg.Text("Demand Remain", size=(12,1))]
            for j in range(len(d2)):
                col_sum = sum(current_solution[i][j] for i in range(len(s2)))
                remd = current_demand[j] - col_sum
                dem_row.append(sg.Text(f"{remd:.2f}", size=(8,1), key=f"rem_dem_{j}"))
            dem_row.append(sg.Text("", size=(12,1)))
            alloc_grid.append(dem_row)

            # Update grid and enable buttons
            window["-GRIDCOL-"].update(alloc_grid)
            window["-MODI-"].update(disabled=False)
            window["-APPLY-"].update(disabled=False)
            window["-EXPORT-"].update(disabled=False)
            window["-STATUS-"].update(f"Solved ({method}). Total cost: {total:.2f}. You may edit allocations then 'Apply Edits' or click 'Optimize (MODI)'.")
            # Render allocation heatmap
            try:
                fig, ax = plt.subplots(figsize=(5,4))
                arr = np.array(current_solution, dtype=float)
                im = ax.imshow(arr, aspect="auto")
                ax.set_xticks(np.arange(len(d2)))
                ax.set_yticks(np.arange(len(s2)))
                ax.set_xticklabels(d2, rotation=45)
                ax.set_yticklabels(s2)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if arr[i,j] != 0:
                            ax.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center", color="white")
                ax.set_title("Allocations heatmap")
                fig.colorbar(im, ax=ax)
                # place canvas in plot column
                window["-PLOTCOL-"].update([[sg.Canvas(key="-CANVAS-")], [sg.Text(f"Total Cost: {total:.2f}", key="-TOTCOST-")]])
                window.refresh()
                canvas = window["-CANVAS-"].TKCanvas
                fig_agg = draw_figure(canvas, fig)
            except Exception:
                pass  # harmless if plotting fails

        if event == "-APPLY-":
            # Read user-edited allocations, validate (non-negative, meet supply/demand), then update solution and cost
            if current_solution is None:
                sg.popup_error("No current solution to edit. Solve a problem first.")
                continue
            m = len(current_sources)
            n = len(current_destinations)
            try:
                new_sol = [[0.0]*n for _ in range(m)]
                for i in range(m):
                    for j in range(n):
                        key = f"alloc_{i}_{j}"
                        v = values.get(key)
                        if v is None or v == "":
                            raise ValueError("All allocation cells must be filled (enter 0 if none).")
                        val = float(v)
                        if val < 0:
                            raise ValueError("Allocations must be non-negative.")
                        new_sol[i][j] = val
                # Validate row sums <= supply and col sums <= demand and totals equal
                row_sums = [sum(new_sol[i]) for i in range(m)]
                col_sums = [sum(new_sol[i][j] for i in range(m)) for j in range(n)]
                for i in range(m):
                    if row_sums[i] - current_supply[i] > 1e-6:
                        raise ValueError(f"Row {current_sources[i]} allocations exceed its supply ({row_sums[i]} > {current_supply[i]}).")
                for j in range(n):
                    if col_sums[j] - current_demand[j] > 1e-6:
                        raise ValueError(f"Column {current_destinations[j]} allocations exceed its demand ({col_sums[j]} > {current_demand[j]}).")
                # If totals don't match, warn and ask user to confirm (we allow partial allocations)
                total_alloc = sum(row_sums)
                if abs(total_alloc - sum(current_supply)) > 1e-6 and abs(total_alloc - sum(current_demand)) > 1e-6:
                    ans = sg.popup_yes_no("Total allocations do not match total supply/demand. Proceed anyway?")
                    if ans != "Yes":
                        continue
                # accept new allocations
                current_solution = new_sol
                current_total_cost = cost_of_solution(current_solution, current_costs)
                sg.popup("Allocations applied", f"New total cost: {current_total_cost:.4f}")
                window["-STATUS-"].update(f"Manual allocations applied. Total cost: {current_total_cost:.2f}")
                window["-TOTCOST-"].update(f"Total Cost: {current_total_cost:.2f}")
                # update remaining labels
                for i in range(m):
                    rem = current_supply[i] - sum(current_solution[i])
                    try:
                        window[f"rem_sup_{i}"].update(f"{rem:.2f}")
                    except Exception:
                        pass
                for j in range(n):
                    remd = current_demand[j] - sum(current_solution[i][j] for i in range(m))
                    try:
                        window[f"rem_dem_{j}"].update(f"{remd:.2f}")
                    except Exception:
                        pass
                # re-plot updated allocations
                try:
                    fig, ax = plt.subplots(figsize=(5,4))
                    arr = np.array(current_solution, dtype=float)
                    im = ax.imshow(arr, aspect="auto")
                    ax.set_xticks(np.arange(n))
                    ax.set_yticks(np.arange(m))
                    ax.set_xticklabels(current_destinations, rotation=45)
                    ax.set_yticklabels(current_sources)
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            if arr[i,j] != 0:
                                ax.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center", color="white")
                    ax.set_title("Allocations heatmap (edited)")
                    fig.colorbar(im, ax=ax)
                    canvas = window["-CANVAS-"].TKCanvas
                    fig_agg = draw_figure(canvas, fig)
                except Exception:
                    pass
            except Exception as e:
                sg.popup_error("Invalid allocations", str(e))

        if event == "-MODI-":
            if current_solution is None:
                sg.popup_error("No solution available. Solve first.")
                continue
            # Run MODI optimization with safety iterations
            try:
                opt_sol, opt_cost, iter_logs = optimize_by_modi(current_solution, current_costs, current_sources, current_destinations, max_iterations=50)
                # Present iteration logs
                lines = []
                for entry in iter_logs:
                    it = entry["iteration"]
                    info = entry["info"]
                    c = entry["cost"]
                    lines.append(f"Iteration {it}: cost {c:.4f}")
                    for ln in info.get("logs", []):
                        lines.append("  - " + ln)
                    entering = info.get("entering")
                    if entering:
                        lines.append(f"    entering cell {entering[0], entering[1]} delta={entering[2]:.4f}")
                    if info.get("optimal"):
                        lines.append("    Reached optimality (no negative reduced costs).")
                sg.popup_scrolled("\n".join(lines), title="MODI Iterations", size=(80, 30))
                # Update state
                current_solution = opt_sol
                current_total_cost = opt_cost
                # Update allocation inputs on grid
                m = len(current_sources)
                n = len(current_destinations)
                for i in range(m):
                    for j in range(n):
                        try:
                            window[f"alloc_{i}_{j}"].update(str(current_solution[i][j]))
                        except Exception:
                            pass
                window["-STATUS-"].update(f"Optimization finished. Cost: {current_total_cost:.2f}")
                window["-TOTCOST-"].update(f"Total Cost: {current_total_cost:.2f}")
                # re-plot
                try:
                    fig, ax = plt.subplots(figsize=(5,4))
                    arr = np.array(current_solution, dtype=float)
                    im = ax.imshow(arr, aspect="auto")
                    ax.set_xticks(np.arange(n))
                    ax.set_yticks(np.arange(m))
                    ax.set_xticklabels(current_destinations, rotation=45)
                    ax.set_yticklabels(current_sources)
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            if arr[i,j] != 0:
                                ax.text(j, i, f"{arr[i,j]:.0f}", ha="center", va="center", color="white")
                    ax.set_title("Allocations heatmap (optimized)")
                    fig.colorbar(im, ax=ax)
                    canvas = window["-CANVAS-"].TKCanvas
                    fig_agg = draw_figure(canvas, fig)
                except Exception:
                    pass
            except Exception as e:
                sg.popup_error("MODI error", str(e))

        if event == "-EXPORT-":
            if current_solution is None:
                sg.popup_error("No result to export. Solve first.")
                continue
            df = solution_to_dataframe(current_sources, current_destinations, current_solution)
            path = sg.popup_get_file("Save result", save_as=True, file_types=(("Excel Files","*.xlsx"), ("CSV Files","*.csv")), default_extension=".xlsx")
            if path:
                try:
                    if path.lower().endswith(".csv"):
                        df.to_csv(path)
                    else:
                        df.to_excel(path, sheet_name="Allocation")
                    sg.popup("Exported", f"Saved to {path}")
                except Exception as e:
                    sg.popup_error("Export failed", str(e))

    window.close()


if __name__ == "__main__":
    main()
