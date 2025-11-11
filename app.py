# app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import solver
import utils

# --- NEW HELPER FUNCTION ---
def parse_cost_value(value_str):
    """Converts a string to a float, handling 'inf', 'infinity', 'M'."""
    val = value_str.strip().lower()
    if val in ['inf', 'infinity', 'm']:
        return np.inf
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Invalid cost value: '{value_str}'. Please use numbers or 'inf'.")

# ... (cost_of_solution, draw_figure are unchanged)
def cost_of_solution(solution, costs):
    return np.sum(np.array(solution) * np.array(costs))
def draw_figure(canvas_frame, figure):
    for widget in canvas_frame.winfo_children(): widget.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas_frame)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    widget.pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

class EnhancedSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Solver v2.6 (Tkinter)")
        self.root.geometry("1100x750")
        self.cost_entries, self.supply_entries, self.demand_entries, self.alloc_entries = [], [], [], []
        self.current_sources, self.current_destinations, self.current_costs = None, None, None
        self.current_supply, self.current_demand, self.current_solution, self.current_total_cost = None, None, None, None
        self.current_initial_solution = None
        self.create_widgets()
        self.create_input_grid()
        self.on_problem_type_change()
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Frame(main_pane, padding=10, width=300); main_pane.add(control_frame, weight=1)
        right_panel = ttk.Frame(main_pane); main_pane.add(right_panel, weight=4)
        lf_type = ttk.LabelFrame(control_frame, text="Problem Type", padding=10); lf_type.pack(fill=tk.X, pady=5)
        self.problem_type_combo = ttk.Combobox(lf_type, values=["Transportation", "Assignment"], state="readonly"); self.problem_type_combo.set("Transportation"); self.problem_type_combo.pack(fill=tk.X)
        self.problem_type_combo.bind("<<ComboboxSelected>>", self.on_problem_type_change)
        lf_dims = ttk.LabelFrame(control_frame, text="Problem Dimensions", padding=10); lf_dims.pack(fill=tk.X, pady=5)
        ttk.Label(lf_dims, text="Sources:").grid(row=0, column=0, sticky='w')
        self.ns_entry = ttk.Entry(lf_dims, width=5); self.ns_entry.insert(0, "3"); self.ns_entry.grid(row=0, column=1, padx=5)
        ttk.Label(lf_dims, text="Destinations:").grid(row=0, column=2, sticky='w')
        self.nd_entry = ttk.Entry(lf_dims, width=5); self.nd_entry.insert(0, "3"); self.nd_entry.grid(row=0, column=3, padx=5)
        ttk.Button(lf_dims, text="Create Grid", command=self.create_input_grid).grid(row=1, column=0, columnspan=4, pady=5)
        lf_actions = ttk.LabelFrame(control_frame, text="Actions", padding=10); lf_actions.pack(fill=tk.X, pady=5)
        ttk.Button(lf_actions, text="Load from File", command=self.load_file).pack(fill=tk.X, pady=2)
        ttk.Button(lf_actions, text="Export Inputs", command=self.export_inputs).pack(fill=tk.X, pady=2)
        self.method_label = ttk.Label(lf_actions, text="Initial Method:"); self.method_label.pack(anchor='w', pady=(10, 2))
        self.method_combo = ttk.Combobox(lf_actions, values=["North-West Corner", "Least Cost", "Vogel (VAM)"], state="readonly"); self.method_combo.set("Vogel (VAM)"); self.method_combo.pack(fill=tk.X, pady=2)
        self.solve_btn = ttk.Button(lf_actions, text="Solve", command=self.solve); self.solve_btn.pack(fill=tk.X, pady=10)
        self.modi_btn = ttk.Button(lf_actions, text="Optimize (MODI)", command=self.optimize_modi, state=tk.DISABLED); self.modi_btn.pack(fill=tk.X, pady=2)
        self.apply_btn = ttk.Button(lf_actions, text="Apply Manual Edits", command=self.apply_edits, state=tk.DISABLED); self.apply_btn.pack(fill=tk.X, pady=2)
        self.export_btn = ttk.Button(lf_actions, text="Export Result", command=self.export_result, state=tk.DISABLED); self.export_btn.pack(fill=tk.X, pady=2)
        self.status_var = tk.StringVar(value="Ready."); ttk.Label(control_frame, textvariable=self.status_var, wraplength=280).pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        self.input_grid_frame = ttk.LabelFrame(right_panel, text="Inputs", padding=10); self.input_grid_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
        self.solution_grid_frame = ttk.LabelFrame(right_panel, text="Solution", padding=10); self.solution_grid_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
        plot_panel = ttk.Frame(right_panel, padding=10); plot_panel.pack(fill=tk.BOTH, expand=True)
        self.total_cost_label = ttk.Label(plot_panel, text="", font=("Segoe UI", 12, "bold")); self.total_cost_label.pack()
        self.canvas_frame = ttk.Frame(plot_panel); self.canvas_frame.pack(fill=tk.BOTH, expand=True)
    def on_problem_type_change(self, event=None):
        is_transportation = self.problem_type_combo.get() == "Transportation"
        state = tk.NORMAL if is_transportation else tk.DISABLED
        for entry in self.supply_entries + self.demand_entries: entry.config(state=state)
        self.method_combo.config(state=state); self.method_label.config(state=state); self.modi_btn.config(state=state)
        if not is_transportation:
            self.ns_entry.config(state=tk.NORMAL); self.nd_entry.delete(0, tk.END); self.nd_entry.insert(0, self.ns_entry.get()); self.nd_entry.config(state=tk.DISABLED)
        else: self.nd_entry.config(state=tk.NORMAL)
        self.create_input_grid()
    def _clear_frame(self, frame):
        for widget in frame.winfo_children(): widget.destroy()
    def create_input_grid(self, costs=None, supply=None, demand=None):
        self._clear_frame(self.input_grid_frame); self.apply_btn.config(state=tk.DISABLED)
        try:
            num_s = int(self.ns_entry.get())
            if self.problem_type_combo.get() == "Assignment":
                self.nd_entry.config(state=tk.NORMAL); self.nd_entry.delete(0, tk.END); self.nd_entry.insert(0, str(num_s)); self.nd_entry.config(state=tk.DISABLED)
            num_d = int(self.nd_entry.get())
        except ValueError: messagebox.showerror("Error", "Sources/Destinations must be integers."); return
        is_transportation = self.problem_type_combo.get() == "Transportation"
        for j in range(num_d): ttk.Label(self.input_grid_frame, text=f"D{j+1}", font="TkDefaultFont 9 bold").grid(row=0, column=j+1)
        if is_transportation: ttk.Label(self.input_grid_frame, text="Supply", font="TkDefaultFont 9 bold").grid(row=0, column=num_d+1)
        self.cost_entries, self.supply_entries = [], []
        for i in range(num_s):
            ttk.Label(self.input_grid_frame, text=f"S{i+1}", font="TkDefaultFont 9 bold").grid(row=i+1, column=0, padx=5)
            row_entries = []
            for j in range(num_d):
                cost_val = costs[i][j] if costs else 0
                display_val = 'inf' if cost_val == np.inf else str(cost_val)
                e = ttk.Entry(self.input_grid_frame, width=8); e.insert(0, display_val); e.grid(row=i+1, column=j+1, padx=2, pady=2); row_entries.append(e)
            self.cost_entries.append(row_entries)
            if is_transportation:
                se = ttk.Entry(self.input_grid_frame, width=8); se.insert(0, str(supply[i] if supply else 0)); se.grid(row=i+1, column=num_d+1, padx=5); self.supply_entries.append(se)
        self.demand_entries = []
        if is_transportation:
            ttk.Label(self.input_grid_frame, text="Demand", font="TkDefaultFont 9 bold").grid(row=num_s+1, column=0)
            for j in range(num_d):
                de = ttk.Entry(self.input_grid_frame, width=8); de.insert(0, str(demand[j] if demand else 0)); de.grid(row=num_s+1, column=j+1); self.demand_entries.append(de)
        self.status_var.set(f"Created {num_s}x{num_d} grid. Enter values.")
    
    def parse_grid_values(self):
        try:
            costs = [[parse_cost_value(e.get()) for e in row] for row in self.cost_entries]
            supply = [float(e.get()) for e in self.supply_entries] if self.supply_entries else None
            demand = [float(e.get()) for e in self.demand_entries] if self.demand_entries else None
            sources = [f"S{i+1}" for i in range(len(costs))]; destinations = [f"D{j+1}" for j in range(len(costs[0]))]
            return sources, destinations, costs, supply, demand
        except ValueError as e: raise ValueError(f"Input Error: {e}")
        
    def solve(self):
        self._clear_frame(self.solution_grid_frame)
        try:
            s, d, c, sup, dem = self.parse_grid_values()
            problem_type = self.problem_type_combo.get()

            if problem_type == "Transportation":
                if sum(sup) != sum(dem): messagebox.showwarning("Unbalanced", "Total Supply != Total Demand.")
                if self.method_combo.get() == "North-West Corner" and np.isinf(np.array(c)).any():
                    messagebox.showwarning("North-West Corner Warning", "This method ignores costs and may allocate to a forbidden (infinite cost) cell.")
                method = self.method_combo.get()
                if method == "North-West Corner": solution, steps = solver.north_west_corner(sup, dem)
                elif method == "Least Cost": solution, steps = solver.least_cost(c, sup, dem)
                else: solution, steps = solver.vogel_approximation(c, sup, dem)
                self.current_initial_solution = solution.copy()
                total_cost = cost_of_solution(solution, c)
                utils.StepsWindow(self.root, f"{method} Steps", steps, s, d)
            else:
                assignments, total_cost, solution, steps = solver.solve_assignment_problem(c)
                utils.StepsWindow(self.root, "Assignment Solver Steps", steps, s, d)
            
            if np.isinf(total_cost):
                messagebox.showerror("Infeasible Solution", "The solution requires using a forbidden (infinite cost) route. This means no feasible solution exists for the given constraints.")
                return

            self.current_sources, self.current_destinations, self.current_costs = s, d, c
            self.current_supply, self.current_demand = sup, dem
            self.current_solution, self.current_total_cost = solution, total_cost
            self.status_var.set(f"Solved. Cost: {total_cost:.2f}.")
            self.create_allocation_grid(); self.update_plot()
            self.apply_btn.config(state=tk.NORMAL); self.export_btn.config(state=tk.NORMAL)
            if problem_type == "Transportation": self.modi_btn.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Error", str(e))

    def optimize_modi(self):
        if self.current_initial_solution is None: messagebox.showerror("Error", "Find an initial solution first via 'Solve'."); return
        try:
            optimal_solution, modi_steps = solver.modi_method(self.current_costs, self.current_initial_solution)
            self.current_solution = optimal_solution
            self.current_total_cost = cost_of_solution(self.current_solution, self.current_costs)
            if np.isinf(self.current_total_cost):
                messagebox.showerror("Infeasible Solution", "MODI resulted in an infinite cost solution. This can happen if the initial solution was invalid (e.g., from NWC on a forbidden route).")
                return
            utils.StepsWindow(self.root, "MODI Optimization Steps", modi_steps, self.current_sources, self.current_destinations)
            self.create_allocation_grid(); self.update_plot()
            self.status_var.set(f"MODI optimization complete. Final Cost: {self.current_total_cost:.2f}")
            messagebox.showinfo("MODI", "Optimization complete! Check the steps window.")
        except Exception as e: messagebox.showerror("MODI Error", f"Could not optimize: {e}")
    
    # ... (rest of the class methods are unchanged)
    def create_allocation_grid(self):
        self._clear_frame(self.solution_grid_frame)
        for j, dest in enumerate(self.current_destinations): ttk.Label(self.solution_grid_frame, text=dest, font="TkDefaultFont 9 bold").grid(row=0, column=j+1)
        self.alloc_entries = []
        for i, src in enumerate(self.current_sources):
            ttk.Label(self.solution_grid_frame, text=src, font="TkDefaultFont 9 bold").grid(row=i+1, column=0, padx=5)
            row_entries = []
            for j in range(len(self.current_destinations)):
                e = ttk.Entry(self.solution_grid_frame, width=8); e.insert(0, f"{self.current_solution[i][j]:.2f}"); e.grid(row=i+1, column=j+1, padx=2, pady=2); row_entries.append(e)
            self.alloc_entries.append(row_entries)
    def update_plot(self):
        self._clear_frame(self.canvas_frame); self.total_cost_label.config(text=f"Total Cost: {self.current_total_cost:.2f}")
        fig, ax = plt.subplots(figsize=(5, 4))
        arr = np.array(self.current_solution, dtype=float)
        vmax = None if self.problem_type_combo.get() == "Transportation" else 1 
        im = ax.imshow(arr, aspect="auto", cmap="Greens", vmax=vmax)
        ax.set_xticks(np.arange(len(self.current_destinations))); ax.set_yticks(np.arange(len(self.current_sources)))
        ax.set_xticklabels(self.current_destinations, rotation=45, ha="right"); ax.set_yticklabels(self.current_sources)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] > 1e-6: ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", color="white")
        ax.set_title("Allocations / Assignments"); fig.tight_layout()
        draw_figure(self.canvas_frame, fig)
    def apply_edits(self):
        try:
            new_sol = [[float(e.get()) for e in row] for row in self.alloc_entries]
            self.current_solution = np.array(new_sol)
            self.current_total_cost = cost_of_solution(self.current_solution, self.current_costs)
            self.status_var.set(f"Manual edits applied. New cost: {self.current_total_cost:.2f}")
            self.update_plot()
        except Exception as e: messagebox.showerror("Error", f"Invalid allocation values: {e}")
    def export_inputs(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            s, d, c, sup, dem = self.parse_grid_values()
            df = pd.DataFrame(c, index=s, columns=d)
            if self.problem_type_combo.get() == "Transportation" and sup and dem:
                df['Supply'] = sup; dem_row = pd.Series(dem, index=d, name='Demand').to_frame().T
                df = pd.concat([df, dem_row])
            df.to_csv(path); messagebox.showinfo("Success", "Input data exported.")
        except Exception as e: messagebox.showerror("Export Error", str(e))
    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Excel", "*.csv;*.xls;*.xlsx")])
        if not path: return
        try:
            s, d, c, sup, dem = utils.import_data(path)
            self.ns_entry.delete(0, tk.END); self.ns_entry.insert(0, str(len(sup)))
            self.nd_entry.delete(0, tk.END); self.nd_entry.insert(0, str(len(dem)))
            self.problem_type_combo.set("Transportation"); self.on_problem_type_change()
            self.create_input_grid(c, sup, dem)
            self.status_var.set(f"Loaded from {os.path.basename(path)}")
        except Exception as e: messagebox.showerror("File Load Error", str(e))
    def export_result(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            df = pd.DataFrame(self.current_solution, index=self.current_sources, columns=self.current_destinations)
            df.to_csv(path); messagebox.showinfo("Success", f"Result saved to {path}")
        except Exception as e: messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedSolverApp(root)
    root.mainloop()

