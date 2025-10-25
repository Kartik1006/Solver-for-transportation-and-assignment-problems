# main.py

import PySimpleGUI as sg
import numpy as np
import solver  # Assume solver.py is in the same directory
import utils  # Assume utils.py is in the same directory

# --- UI Theme and Styling ---
sg.theme('DarkGrey9')  # A modern, clean dark theme
BUTTON_STYLE = {'font': ('Helvetica', 12), 'size': (20, 1), 'button_color': ('#FFFFFF', '#0078D4')}
HEADER_FONT = ('Helvetica', 20, 'bold')
SUBHEADER_FONT = ('Helvetica', 14)
BODY_FONT = ('Helvetica', 12)
STATUS_FONT = ('Helvetica', 10)


# --- UI Layout Definitions ---

def create_main_window():
    """Creates the main application window with input and solve tabs."""

    # --- Input Tab Layout ---
    manual_entry_col = [
        [sg.Text('Supply per Source', font=SUBHEADER_FONT)],
        [sg.Multiline(default_text='50\n70\n30', size=(15, 5), key='-SUPPLY-', font=BODY_FONT,
                      tooltip="Enter one supply value per line")],
        [sg.Text('Demand per Destination', font=SUBHEADER_FONT)],
        [sg.Input(default_text='20 60 70', size=(25, 1), key='-DEMAND-', font=BODY_FONT,
                  tooltip="Enter demand values separated by spaces")],
    ]

    cost_matrix_col = [
        [sg.Text('Cost Matrix (rows of space-separated values)', font=SUBHEADER_FONT)],
        [sg.Multiline(default_text='16 18 21\n12 14 17\n19 20 10', size=(30, 8), key='-COSTS-', font=BODY_FONT)]
    ]

    file_import_frame = sg.Frame('Load from File', [
        [sg.Input(key='-FILE_PATH-', visible=False, enable_events=True)],
        [sg.FileBrowse('Import CSV / Excel', **BUTTON_STYLE, target='-FILE_PATH-'),
         sg.Text("", key='-FILE_NAME-', font=BODY_FONT)]
    ], font=SUBHEADER_FONT)

    input_tab_layout = [
        [sg.Column(manual_entry_col, vertical_alignment='top'), sg.VSeperator(), sg.Column(cost_matrix_col)],
        [sg.HorizontalSeparator(pad=(0, 10))],
        [file_import_frame]
    ]

    # --- Solve Tab Layout ---
    methods_frame = sg.Frame('Configuration', [
        [sg.Text("1. Select Initial Solution Method", font=SUBHEADER_FONT)],
        [sg.Radio("North-West Corner", "METHOD", default=True, key='-NWC-', font=BODY_FONT)],
        [sg.Radio("Least Cost", "METHOD", key='-LCM-', font=BODY_FONT)],
        [sg.Radio("Vogel's Approximation (VAM)", "METHOD", key='-VAM-', font=BODY_FONT)],
        [sg.HorizontalSeparator(pad=(0, 10))],
        [sg.Text("2. Optimize the Solution", font=SUBHEADER_FONT)],
        [sg.Checkbox("Find Optimal Solution using MODI Method", default=True, key='-OPTIMIZE-', font=BODY_FONT)]
    ], font=SUBHEADER_FONT, element_justification='center')

    solve_button_layout = [
        [sg.VPush()],
        [sg.Push(), sg.Button('✨ Solve Transportation Problem', button_color=('#FFFFFF', '#009933'),
                              font=('Helvetica', 14, 'bold'), size=(30, 2), key='-SOLVE-'), sg.Push()],
        [sg.VPush()]
    ]

    solve_tab_layout = [
        [methods_frame],
        [sg.Column(solve_button_layout)]
    ]

    # --- Main Layout with Tabs ---
    layout = [
        [sg.Text('Transportation Problem Optimization Tool', font=HEADER_FONT, justification='center', pad=(0, 10))],
        [sg.TabGroup([
            [sg.Tab('Step 1: Input Data', input_tab_layout),
             sg.Tab('Step 2: Solve', solve_tab_layout)]
        ], font=SUBHEADER_FONT)],
        [sg.StatusBar("", size=(80, 1), key='-STATUS-', font=STATUS_FONT)]
    ]

    return sg.Window('Transportation Solver v2.0', layout, finalize=True)


def create_results_window(solution_data, total_cost, sources, destinations, costs):
    """Creates the results window, generated dynamically after solving."""

    solution_headings = ['Source ↓ | Destination →'] + destinations
    solution_table_data = [[sources[i]] + [f"{val:.0f}" for val in row] for i, row in enumerate(solution_data)]

    results_layout = [
        [sg.Text('Optimal Solution Found', font=HEADER_FONT)],
        [sg.Text(f'Total Transportation Cost: ${total_cost:,.2f}', font=('Helvetica', 16, 'bold'),
                 text_color='#77dd77')],
        [sg.HorizontalSeparator(pad=(0, 10))],
        [
            sg.Column([
                [sg.Text('Final Allocation Matrix', font=SUBHEADER_FONT)],
                [sg.Table(values=solution_table_data, headings=solution_headings, auto_size_columns=True,
                          justification='center', num_rows=min(10, len(sources)), key='-SOLUTION_TABLE-',
                          font=BODY_FONT,
                          header_font=('Helvetica', 12, 'bold'), header_background_color='#404040')],
                [sg.Input(key='-EXPORT_PATH-', visible=False, enable_events=True)],
                [sg.FileSaveAs('Export Results to CSV', **BUTTON_STYLE, target='-EXPORT_PATH-',
                               file_types=(("CSV Files", "*.csv"),))]
            ], vertical_alignment='top'),
            sg.VSeperator(),
            sg.Column([
                [sg.Text('Shipment Visualization', font=SUBHEADER_FONT)],
                [sg.Canvas(key='-CANVAS-')]
            ])
        ],
        [sg.Push(), sg.Button('Close', font=('Helvetica', 12), size=(10, 1)), sg.Push()]
    ]

    window = sg.Window('Solution Results', results_layout, finalize=True, modal=True)

    fig = utils.create_allocation_heatmap(solution_data, sources, destinations)
    utils.draw_figure(window['-CANVAS-'].TKCanvas, fig)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):
            break
        if event == '-EXPORT_PATH-':
            filepath = values['-EXPORT_PATH-']
            if filepath:
                utils.export_data(filepath, sources, destinations, solution_data, total_cost)
                sg.popup_quick_message('Exported successfully!', font=BODY_FONT, auto_close_duration=2)

    window.close()


def main():
    """Main application event loop."""
    main_window = create_main_window()

    while True:
        event, values = main_window.read()
        if event == sg.WIN_CLOSED:
            break

        try:
            main_window['-STATUS-'].update('')

            if event == '-FILE_PATH-':
                filepath = values['-FILE_PATH-']
                sources, dests, costs, supply, demand = utils.import_data(filepath)

                supply_str, demand_str, costs_str = "\n".join(map(str, supply)), " ".join(map(str, demand)), "\n".join(
                    " ".join(map(str, row)) for row in costs)

                main_window['-SUPPLY-'].update(supply_str)
                main_window['-DEMAND-'].update(demand_str)
                main_window['-COSTS-'].update(costs_str)
                main_window['-FILE_NAME-'].update(filepath.split('/')[-1])
                main_window['-STATUS-'].update('Data successfully imported.')

            if event == '-SOLVE-':
                supply = [int(s) for s in values['-SUPPLY-'].strip().split('\n') if s]
                demand = [int(d) for d in values['-DEMAND-'].strip().split() if d]
                costs = [[int(c) for c in row.split() if c] for row in values['-COSTS-'].strip().split('\n') if row]

                if not all([supply, demand, costs]):
                    raise ValueError("Input fields cannot be empty.")
                if len(costs) != len(supply) or any(len(row) != len(demand) for row in costs):
                    raise ValueError("Dimension mismatch: Check cost matrix rows/columns against supply/demand counts.")

                if sum(supply) != sum(demand):
                    sg.popup_notify("Warning: Problem is unbalanced (Total Supply != Total Demand).",
                                    title="Unbalanced Problem", font=BODY_FONT)

                if values['-NWC-']:
                    initial_solution = solver.north_west_corner(supply, demand)
                elif values['-LCM-']:
                    initial_solution = solver.least_cost(costs, supply, demand)
                else:
                    initial_solution = solver.vogel_approximation(costs, supply, demand)

                final_solution = solver.modi_method(costs, initial_solution) if values[
                    '-OPTIMIZE-'] else initial_solution
                total_cost = np.sum(np.array(final_solution) * np.array(costs))

                num_sources, num_dests = final_solution.shape
                sources = [f'Source {i + 1}' for i in range(num_sources)]
                destinations = [f'Destination {j + 1}' for j in range(num_dests)]

                create_results_window(final_solution, total_cost, sources, destinations, costs)

        except Exception as e:
            main_window['-STATUS-'].update(f'Error: {e}')
            sg.popup_error(f'An error occurred:\n\n{e}', title='Error', font=BODY_FONT)

    main_window.close()
if __name__ == '__main__':
    main()