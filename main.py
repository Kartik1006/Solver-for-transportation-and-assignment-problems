# main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
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

# --- UI Theme and Styling ---
HEADER_FONT = ('Helvetica', 20, 'bold')
SUBHEADER_FONT = ('Helvetica', 14)
BODY_FONT = ('Helvetica', 12)
STATUS_FONT = ('Helvetica', 10)

# ... (ResultsWindow class is unchanged and correct)
class ResultsWindow(tk.Toplevel):
    def __init__(self, parent, problem_type, solution_data, total_cost, sources, destinations, costs):
        super().__init__(parent)
        self.problem_type, self.solution_data, self.total_cost = problem_type, solution_data, total_cost
        self.sources, self.destinations, self.costs = sources, destinations, costs
        self.title("Solution Results"); self.transient(parent); self.grab_set()
        self.configure(padx=15, pady=15); self.create_widgets(); self.protocol("WM_DELETE_WINDOW", self.close)
    def create_widgets(self):
        ttk.Label(self, text='Optimal Solution Found', font=HEADER_FONT).pack(pady=(0, 10))
        ttk.Label(self, text=f'Minimum Total Cost: ${self.total_cost:,.2f}', font=('Helvetica', 16, 'bold'), foreground='#009933').pack(pady=(0, 10))
        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=10)
        main_frame = ttk.Frame(self); main_frame.pack(fill='both', expand=True)
        left_col = ttk.Frame(main_frame); left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        if self.problem_type == "Transportation": self.display_transportation_results(left_col)
        else: self.display_assignment_results(left_col)
        ttk.Button(left_col, text='Export Results to CSV', command=self.export_results).pack(pady=10)
        ttk.Separator(main_frame, orient='vertical').pack(side='left', fill='y', padx=5)
        right_col = ttk.Frame(main_frame); right_col.pack(side='right', fill='both', expand=True, padx=(10, 0))
        ttk.Label(right_col, text='Solution Visualization', font=SUBHEADER_FONT).pack(anchor='w')
        canvas_frame = ttk.Frame(right_col, relief="sunken", borderwidth=1); canvas_frame.pack(fill="both", expand=True, pady=5)
        heatmap_data = self.solution_data[2] if self.problem_type == "Assignment" else self.solution_data
        fig = utils.create_allocation_heatmap(heatmap_data, self.sources, self.destinations)
        self.canvas_widget = utils.draw_figure(canvas_frame, fig).get_tk_widget(); self.canvas_widget.pack(fill="both", expand=True)
        ttk.Button(self, text='Close', command=self.close).pack(pady=(20, 0))
    def display_transportation_results(self, parent):
        ttk.Label(parent, text='Final Allocation Matrix', font=SUBHEADER_FONT).pack(anchor='w')
        tree = ttk.Treeview(parent, columns=['Source ↓'] + self.destinations, show='headings')
        for col in ['Source ↓'] + self.destinations: tree.heading(col, text=col); tree.column(col, anchor='center', width=100)
        for i, row_data in enumerate(self.solution_data): tree.insert('', 'end', values=[self.sources[i]] + [f"{val:.0f}" for val in row_data])
        tree.pack(fill='both', expand=True, pady=5)
    def display_assignment_results(self, parent):
        ttk.Label(parent, text='Optimal Assignments', font=SUBHEADER_FONT).pack(anchor='w')
        assignments, _, _ = self.solution_data
        text_widget = tk.Text(parent, height=len(assignments), font=BODY_FONT, relief="flat", background=self.cget('bg'))
        text_widget.pack(fill='x', pady=5)
        for r, c in assignments: text_widget.insert(tk.END, f"{self.sources[r]}  ->  {self.destinations[c]}\n")
        text_widget.config(state="disabled")
    def export_results(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        try:
            if self.problem_type == "Transportation": df = pd.DataFrame(self.solution_data, index=self.sources, columns=self.destinations)
            else: df = pd.DataFrame({'Source': [self.sources[r] for r,c in self.solution_data[0]], 'Assigned_To': [self.destinations[c] for r,c in self.solution_data[0]]})
            df.to_csv(filepath); messagebox.showinfo("Success", "Exported successfully!", parent=self)
        except Exception as e: messagebox.showerror("Error", f"Could not export file:\n{e}", parent=self)
    def close(self): self.destroy()

class TransportationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Optimization Solver v2.6 (Tkinter)')
        self.root.minsize(550, 500)
        self.problem_type_var, self.supply_var, self.demand_var, self.costs_var = tk.StringVar(value="Transportation"), tk.StringVar(value='50\n70\n30'), tk.StringVar(value='20 60 70'), tk.StringVar(value='16 18 M\n12 14 17\n19 20 10')
        self.method_var, self.optimize_var, self.file_name_var, self.status_var = tk.StringVar(value='NWC'), tk.BooleanVar(value=True), tk.StringVar(value=""), tk.StringVar(value="")
        self.create_widgets(); self.on_problem_type_change()
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill="both", expand=True)
        ttk.Label(main_frame, text='Optimization Solver', font=HEADER_FONT, anchor="center").pack(fill='x', pady=(0, 10))
        type_frame = ttk.LabelFrame(main_frame, text="Problem Type", padding="10"); type_frame.pack(fill='x', pady=5)
        ttk.Radiobutton(type_frame, text="Transportation", variable=self.problem_type_var, value="Transportation", command=self.on_problem_type_change).pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="Assignment (Square Matrix)", variable=self.problem_type_var, value="Assignment", command=self.on_problem_type_change).pack(side='left', padx=10)
        notebook = ttk.Notebook(main_frame); notebook.pack(fill="both", expand=True, pady=5)
        input_tab, solve_tab = ttk.Frame(notebook, padding="10"), ttk.Frame(notebook, padding="10")
        notebook.add(input_tab, text='Step 1: Input Data'); notebook.add(solve_tab, text='Step 2: Solve')
        self.create_input_tab(input_tab); self.create_solve_tab(solve_tab)
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w', font=STATUS_FONT, padding="2 5"); status_bar.pack(side='bottom', fill='x')
    def on_problem_type_change(self):
        is_transportation = self.problem_type_var.get() == "Transportation"
        state = tk.NORMAL if is_transportation else tk.DISABLED
        self.supply_text.config(state=state); self.demand_entry.config(state=state); self.supply_label.config(state=state); self.demand_label.config(state=state)
        for child in self.methods_frame.winfo_children():
            if isinstance(child, (ttk.Radiobutton, ttk.Checkbutton, ttk.Label, ttk.Separator)): child.config(state=state)
    def create_input_tab(self, parent):
        top_frame = ttk.Frame(parent); top_frame.pack(fill='x', expand=True)
        manual_entry_col = ttk.Frame(top_frame); manual_entry_col.pack(side='left', fill='y', padx=(0, 10))
        self.supply_label = ttk.Label(manual_entry_col, text='Supply per Source', font=SUBHEADER_FONT); self.supply_label.pack(anchor='w', pady=(0, 5))
        self.supply_text = tk.Text(manual_entry_col, width=20, height=5, font=BODY_FONT); self.supply_text.insert('1.0', self.supply_var.get()); self.supply_text.pack(anchor='w', pady=(0, 10))
        self.demand_label = ttk.Label(manual_entry_col, text='Demand per Destination', font=SUBHEADER_FONT); self.demand_label.pack(anchor='w', pady=(0, 5))
        self.demand_entry = ttk.Entry(manual_entry_col, textvariable=self.demand_var, width=30, font=BODY_FONT); self.demand_entry.pack(anchor='w')
        ttk.Separator(top_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        cost_matrix_col = ttk.Frame(top_frame); cost_matrix_col.pack(side='left', fill='both', expand=True)
        ttk.Label(cost_matrix_col, text='Cost Matrix', font=SUBHEADER_FONT).pack(anchor='w', pady=(0, 5))
        self.costs_text = tk.Text(cost_matrix_col, width=35, height=8, font=BODY_FONT); self.costs_text.insert('1.0', self.costs_var.get()); self.costs_text.pack(anchor='w', fill='both', expand=True)
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=20)
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding="10"); file_frame.pack(fill='x')
        ttk.Button(file_frame, text="Import Data", command=self.import_from_file).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Export Inputs", command=self.export_inputs).pack(side='left', padx=5)
        ttk.Label(file_frame, textvariable=self.file_name_var, font=BODY_FONT).pack(side='left', padx=10)
    def create_solve_tab(self, parent):
        self.methods_frame = ttk.LabelFrame(parent, text="Transportation Configuration", padding="15"); self.methods_frame.pack(fill='x', pady=10)
        ttk.Label(self.methods_frame, text="1. Select Initial Solution Method", font=SUBHEADER_FONT).pack(anchor='w', pady=(0, 10))
        ttk.Radiobutton(self.methods_frame, text="North-West Corner", variable=self.method_var, value='NWC').pack(anchor='w')
        ttk.Radiobutton(self.methods_frame, text="Least Cost", variable=self.method_var, value='LCM').pack(anchor='w')
        ttk.Radiobutton(self.methods_frame, text="Vogel's Approximation (VAM)", variable=self.method_var, value='VAM').pack(anchor='w')
        ttk.Separator(self.methods_frame, orient='horizontal').pack(fill='x', pady=15)
        ttk.Label(self.methods_frame, text="2. Optimize the Solution", font=SUBHEADER_FONT).pack(anchor='w', pady=(0, 10))
        ttk.Checkbutton(self.methods_frame, text="Find Optimal Solution using MODI Method", variable=self.optimize_var).pack(anchor='w')
        solve_button = ttk.Button(parent, text='✨ Solve Problem', command=self.solve, style="Accent.TButton"); solve_button.pack(pady=30, ipady=10)
        style = ttk.Style(); style.configure("Accent.TButton", font=('Helvetica', 14, 'bold'))
    
    def get_inputs(self):
        costs_str = self.costs_text.get('1.0', tk.END).strip()
        costs = [[parse_cost_value(c) for c in row.split() if c] for row in costs_str.split('\n') if row]
        if self.problem_type_var.get() == "Transportation":
            supply = [float(s) for s in self.supply_text.get('1.0', tk.END).strip().split('\n') if s]
            demand = [float(d) for d in self.demand_var.get().strip().split() if d]
            return costs, supply, demand
        return costs, None, None
        
    def export_inputs(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        try:
            costs, supply, demand = self.get_inputs()
            if not costs: raise ValueError("Cost matrix is empty.")
            df = pd.DataFrame(costs, index=[f'Source_{i+1}' for i in range(len(costs))], columns=[f'Dest_{j+1}' for j in range(len(costs[0]))])
            if self.problem_type_var.get() == "Transportation" and supply and demand:
                df['Supply'] = supply
                demand_row = pd.Series(demand, index=df.columns[:-1], name='Demand').to_frame().T
                df = pd.concat([df, demand_row])
            df.to_csv(filepath); messagebox.showinfo("Success", "Input data exported successfully!")
        except Exception as e: messagebox.showerror("Export Error", f"Could not export inputs:\n{e}")
    def import_from_file(self):
        filepath = filedialog.askopenfilename(title="Select a data file", filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")])
        if not filepath: return
        try:
            self.status_var.set(''); _, _, costs, supply, demand = utils.import_data(filepath)
            self.problem_type_var.set("Transportation"); self.on_problem_type_change()
            self.supply_text.delete('1.0', tk.END); self.supply_text.insert('1.0', "\n".join(map(str, supply)))
            self.demand_var.set(" ".join(map(str, demand)))
            costs_str = "\n".join(" ".join(str(c) if c != np.inf else 'inf' for c in r) for r in costs)
            self.costs_text.delete('1.0', tk.END); self.costs_text.insert('1.0', costs_str)
            self.file_name_var.set(filepath.split('/')[-1]); self.status_var.set('Data successfully imported.')
        except Exception as e: self.status_var.set(f'Error: {e}'); messagebox.showerror('Import Error', f'An error occurred:\n\n{e}')

    def solve(self):
        try:
            self.status_var.set('')
            problem_type = self.problem_type_var.get()
            costs, supply, demand = self.get_inputs()

            if not costs: raise ValueError("Cost matrix cannot be empty.")
            num_sources, num_dests = len(costs), len(costs[0])
            sources, destinations = [f'S{i+1}' for i in range(num_sources)], [f'D{j+1}' for j in range(num_dests)]

            if problem_type == "Transportation":
                if not all([supply, demand]): raise ValueError("Supply/Demand fields cannot be empty.")
                if len(costs) != len(supply) or any(len(row) != len(demand) for row in costs): raise ValueError("Dimension mismatch.")
                if sum(supply) != sum(demand): messagebox.showwarning("Unbalanced", "Total Supply != Total Demand.")
                
                if self.method_var.get() == 'NWC' and np.isinf(np.array(costs)).any():
                    messagebox.showwarning("North-West Corner Warning",
                                           "North-West Corner method ignores costs and may allocate to a forbidden (infinite cost) cell. The result might be infeasible.")

                method = self.method_var.get()
                if method == 'NWC': initial_solution, steps = solver.north_west_corner(supply, demand)
                elif method == 'LCM': initial_solution, steps = solver.least_cost(costs, supply, demand)
                else: initial_solution, steps = solver.vogel_approximation(costs, supply, demand)
                
                final_solution = initial_solution
                if self.optimize_var.get():
                    final_solution, modi_steps = solver.modi_method(costs, initial_solution)
                    steps.extend(modi_steps)

                total_cost = np.sum(np.array(final_solution) * np.array(costs))
                if np.isinf(total_cost):
                    messagebox.showerror("Infeasible Solution", "The final solution contains allocations on forbidden (infinite cost) routes, making the total cost infinite. This can happen if the problem is structured such that no feasible solution exists, or if North-West Corner was used on a problem with forbidden routes.")
                    return

                utils.StepsWindow(self.root, "Transportation Solver Steps", steps, sources, destinations)
                ResultsWindow(self.root, "Transportation", final_solution, total_cost, sources, destinations, costs)

            else: # Assignment
                if num_sources != num_dests: raise ValueError("Assignment problem requires a square cost matrix.")
                assignments, total_cost, solution_matrix, steps = solver.solve_assignment_problem(costs)
                if np.isinf(total_cost):
                     messagebox.showerror("Infeasible Solution", "No feasible assignment could be found without using a forbidden (infinite cost) route.")
                     return
                utils.StepsWindow(self.root, "Assignment Solver Steps", steps, sources, destinations)
                ResultsWindow(self.root, "Assignment", (assignments, total_cost, solution_matrix), total_cost, sources, destinations, costs)

        except Exception as e:
            self.status_var.set(f'Error: {e}')
            messagebox.showerror('Error', f'An error occurred:\n\n{e}')

def main():
    root = tk.Tk()
    app = TransportationSolverApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
