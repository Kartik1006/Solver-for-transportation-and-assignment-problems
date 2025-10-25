# Optimization Solver for Transportation & Assignment Problems

A desktop application built with Python and Tkinter that provides a user-friendly graphical interface for solving two classic operations research problems: the **Transportation Problem** and the **Assignment Problem**.

This tool is designed for students, educators, and professionals who need to solve and visualize these optimization problems. It features an interactive, step-by-step visualizer that breaks down each algorithm, making it an excellent learning aid.

## Key Features

-   **Dual Solvers**: Seamlessly switch between solving Transportation and Assignment problems.
-   **Multiple Algorithms**:
    -   **Transportation Problem**:
        -   Initial Solution: North-West Corner, Least Cost Method, and Vogel's Approximation Method (VAM).
        -   Optimization: Modified Distribution (MODI) method to find the optimal solution.
    -   **Assignment Problem**:
        -   Optimal Solution: Hungarian method (via `scipy`) for cost minimization.
-   **Interactive Step-by-Step Visualizer**:
    -   Walk through algorithms one step at a time with "Next" and "Previous" buttons.
    -   View the state of the allocation matrix at each step.
    -   Key cells are highlighted to show what the algorithm is focusing on.
    -   Clear, contextual descriptions explain each action.
-   **Special Case Handling**:
    -   Define forbidden routes in the cost matrix by entering `inf`, `infinity`, or `M`. The solvers will correctly interpret these as infinite costs and avoid them.
-   **Rich User Interface**:
    -   Intuitive grid-based data entry for costs, supply, and demand.
    -   Dynamic UI that adapts based on the selected problem type.
    -   Final solution is displayed in an editable grid and visualized as a heatmap.
-   **Data Management**:
    -   Import problem data directly from CSV or Excel files.
    -   Export your input data or the final solution to a CSV file.

## Screenshots

Here are some previews of the application in action.

**Main Application (`app.py`)**
<img width="1037" height="732" alt="image" src="https://github.com/user-attachments/assets/9cfcef6f-4ead-4c6a-96f0-02bc4eb677ba" />

**Interactive Step-by-Step Solver**
<img width="1002" height="663" alt="image" src="https://github.com/user-attachments/assets/22a42563-deeb-4d45-8596-9b1b0f8efe16" />

**Final Solution and Visualization**
<img width="1144" height="806" alt="image" src="https://github.com/user-attachments/assets/1923827b-e2d9-4c8a-beca-7d3951506edf" />


## Installation

To run this application on your local machine, follow these steps.

**1. Prerequisites**
-   Python 3.7 or newer.

**2. Clone the Repository**
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

**3. Install Dependencies**
This project uses several scientific and data manipulation libraries. You can install them using pip:
```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl
```

## How to Use

The repository contains two main applications:
-   `app.py`: The primary, feature-rich application with a single-window interface.
-   `main.py`: A simpler, alternative application with a tabbed interface.

**To run the enhanced application (recommended):**
```bash
python app.py
```

**Workflow:**
1.  **Select Problem Type**: Choose "Transportation" or "Assignment" from the top-left dropdown. The UI will adapt accordingly.
2.  **Set Dimensions**: Enter the number of sources and destinations and click "Create Grid". For assignment problems, the grid will always be square.
3.  **Enter Data**:
    -   Fill in the cost matrix in the "Inputs" grid.
    -   For forbidden routes, enter `inf`, `infinity`, or `M`.
    -   If solving a transportation problem, fill in the "Supply" and "Demand" values.
    -   Alternatively, click **"Load from File"** to import data from a CSV or Excel file.
4.  **Solve**:
    -   For a transportation problem, select an initial solution method (e.g., VAM).
    -   Click the **"Solve"** button.
5.  **Review Steps**:
    -   The interactive step-by-step solver window will appear. Use the "Next" and "Previous" buttons to see how the algorithm works.
    -   Close the steps window when you are done.
6.  **View Results**:
    -   The main window now displays the final solution in a grid and as a heatmap.
7.  **Optimize (Optional)**:
    -   If you solved a transportation problem, click **"Optimize (MODI)"** to find the optimal solution. A new steps window for the MODI method will appear.
8.  **Export**:
    -   Use **"Export Inputs"** or **"Export Result"** to save your data to a CSV file.

## File Structure

```
.
├── app.py              # The main, enhanced GUI application.
├── main.py             # A simpler, alternative GUI application.
├── solver.py           # Contains all the backend logic for the optimization algorithms.
├── utils.py            # Helper functions for file I/O, plotting, and the StepsWindow UI class.
└── README.md           # This file.
```

## Technologies Used

-   **Python 3**: Core programming language.
-   **Tkinter**: Python's standard GUI library for the user interface.
-   **NumPy**: For efficient numerical operations and matrix manipulation.
-   **Pandas**: For easy data handling, especially for CSV/Excel import and export.
-   **SciPy**: Used for its optimized implementation of the Hungarian method for the assignment problem.
-   **Matplotlib & Seaborn**: For creating the heatmap visualization of the final solution.
