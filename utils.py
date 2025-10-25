# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def import_data(file_path):
    """Imports data from CSV or Excel files."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path, index_col=0)
    else:
        raise ValueError("Unsupported file format.")

    supply = df.iloc[:-1, -1].values.tolist()
    demand = df.iloc[-1, :-1].values.tolist()
    costs = df.iloc[:-1, :-1].values.tolist()
    sources = df.index[:-1].tolist()
    destinations = df.columns[:-1].tolist()

    return sources, destinations, costs, supply, demand


def export_data(file_path, sources, destinations, solution, total_cost):
    """Exports the solution to a CSV file."""
    df = pd.DataFrame(solution, index=sources, columns=destinations)
    df.loc['Total Cost'] = ''
    df.iloc[-1, 0] = total_cost
    df.to_csv(file_path)


def draw_figure(canvas, figure):
    """Embeds a matplotlib figure in a Tkinter canvas."""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def create_allocation_heatmap(solution, sources, destinations):
    """Creates a heatmap of the final allocations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(solution, annot=True, fmt=".0f", cmap="Greens", linewidths=.5,
                xticklabels=destinations, yticklabels=sources, ax=ax, cbar=False)
    ax.set_title("Optimal Shipment Allocations", fontsize=16)
    plt.tight_layout()
    return fig