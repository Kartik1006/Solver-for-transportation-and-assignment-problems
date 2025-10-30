# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import numpy as np

def import_data(file_path):
    """Imports data from CSV or Excel, converting 'inf', 'infinity', 'M' to np.inf."""
    if file_path.endswith('.csv'): df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(('.xls', '.xlsx')): df = pd.read_excel(file_path, index_col=0)
    else: raise ValueError("Unsupported file format.")
    
    # Replace string representations of infinity with np.inf before processing
    df.replace(['inf', 'infinity', 'M', 'Inf', 'Infinity'], np.inf, inplace=True)

    supply = df.iloc[:-1, -1].astype(float).values.tolist()
    demand = df.iloc[-1, :-1].astype(float).values.tolist()
    costs = df.iloc[:-1, :-1].astype(float).values.tolist()
    sources = df.index[:-1].tolist()
    destinations = df.columns[:-1].tolist()
    return sources, destinations, costs, supply, demand

# ... (export_data, draw_figure, create_allocation_heatmap are unchanged)
def export_data(file_path, sources, destinations, solution, total_cost):
    df = pd.DataFrame(solution, index=sources, columns=destinations)
    df.loc['Total Cost'] = ''
    df.iloc[-1, 0] = total_cost
    df.to_csv(file_path)
def draw_figure(canvas, figure):
    for widget in canvas.winfo_children(): widget.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
def create_allocation_heatmap(solution, sources, destinations):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(solution, annot=True, fmt=".0f", cmap="cividis", linewidths=.5,
                xticklabels=destinations, yticklabels=sources, ax=ax, cbar=True)
    ax.set_title("Optimal Shipment Allocations", fontsize=16)
    plt.tight_layout()
    return fig

class StepsWindow(tk.Toplevel):
    # ... (create_widgets is mostly unchanged)
    def __init__(self, parent, title, steps_data, sources, destinations):
        super().__init__(parent)
        self.title(title); self.transient(parent); self.grab_set()
        self.geometry("800x500")
        self.steps_data, self.sources, self.destinations = steps_data, sources, destinations
        self.current_step_index = 0
        self.create_widgets()
        self.update_display()
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL); main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        matrix_frame = ttk.LabelFrame(main_pane, text="Matrix View", padding=10); main_pane.add(matrix_frame, weight=2)
        cols = [""] + self.destinations
        self.tree = ttk.Treeview(matrix_frame, columns=cols, show="headings")
        for col in cols: self.tree.heading(col, text=col); self.tree.column(col, width=60, anchor='center')
        self.tree.column(cols[0], width=80, anchor='w')
        self.tree.pack(fill=tk.BOTH, expand=True)
        desc_frame = ttk.LabelFrame(main_pane, text="Step Description", padding=10); main_pane.add(desc_frame, weight=1)
        self.desc_label = ttk.Label(desc_frame, text="", wraplength=250, font=("Helvetica", 11)); self.desc_label.pack(fill=tk.BOTH, expand=True)
        nav_frame = ttk.Frame(self, padding=10); nav_frame.pack(fill=tk.X)
        self.prev_button = ttk.Button(nav_frame, text="<< Previous", command=self.prev_step); self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(nav_frame, text="Next >>", command=self.next_step); self.next_button.pack(side=tk.RIGHT)
        self.step_label = ttk.Label(nav_frame, text=""); self.step_label.pack(side=tk.BOTTOM)
        self.tree.tag_configure('highlight', background='#3498db', foreground='white'); self.tree.tag_configure('highlight_plus', background='#2ecc71', foreground='white'); self.tree.tag_configure('highlight_minus', background='#e74c3c', foreground='white')

    def update_display(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        step_data = self.steps_data[self.current_step_index]
        matrix = step_data["matrix"]
        
        for i, row_data in enumerate(matrix):
            # UPDATE: Format numbers and handle infinity
            def format_val(v):
                if v == np.inf: return "∞"
                if v == 0: return "0"
                return f"{v:.2f}".rstrip('0').rstrip('.')

            row_values = [self.sources[i]] + [format_val(val) for val in row_data]
            iid = self.tree.insert("", "end", values=row_values)
            
            for j, _ in enumerate(row_data):
                tags = []
                if step_data.get("highlight_plus") and (i, j) in step_data["highlight_plus"]: tags.append('highlight_plus')
                elif step_data.get("highlight_minus") and (i, j) in step_data["highlight_minus"]: tags.append('highlight_minus')
                elif step_data.get("highlight") and (i, j) in step_data["highlight"]: tags.append('highlight')
                if tags: self.tree.item(iid, tags=tuple(tags))

        self.desc_label.config(text=step_data["description"])
        self.step_label.config(text=f"Step {self.current_step_index + 1} of {len(self.steps_data)}")
        self.prev_button.config(state=tk.NORMAL if self.current_step_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_step_index < len(self.steps_data) - 1 else tk.DISABLED)

    def next_step(self):
        if self.current_step_index < len(self.steps_data) - 1: self.current_step_index += 1; self.update_display()
    def prev_step(self):
        if self.current_step_index > 0: self.current_step_index -= 1; self.update_display()
