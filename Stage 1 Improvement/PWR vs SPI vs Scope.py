import os
import sys
import inspect
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import log_spi_or_scope
import log_file
from matplotlib import pyplot

from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools
_P = pyplot.plot


# ## txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# ## Chrome action for plotly:
plot_offline = True
kill_chrome = False and plot_offline
auto_open_chrome_output = True and plot_offline
# ## Inverter stuff:
inverter_type = 'Venus3'
log_with_time_stamp = True
# ## Test type:
# ## 1 = Regular Arc search (via State Machine).
# ## 2 = False Alarms (no Arc search).
# ##3 = False Alarms, BUT, records will be cut manually when the state machine is = 10.
# ## 4 = No Filter.
test_type = 4
# ## divide the number of log per plot (in other words, records in each html output):
plots_per_test = 1
# ## Folders and filters:
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\V-DC Arcs 01 (01-07-2021)'
# ## Leave empty if you would like to skip:
path_logs_string_filter = 'Rec001'
# ## plot name ('.html' ass added later):
plot_name = f'PWR vs. SPI vs. Scope Records'
# ## divide the output plots into 2D figures:
plot_columns = 1
# ## Energy Rise and Current Drop parameters:
energy_rise_th_steps = np.array([15, 12, 10, 8, 6])    # Best = np.arange(6, 17, 1)
current_drop_th_steps = np.arange(0.2, 0.0, -0.1)      # Best = np.arange(0.02, 0.22, 0.02)
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
# ## Voltage Drop parameters:
voltage_drop_th_steps = np.arange(0.0, 0.5, 0.1)
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 5
filter_size_2 = 5
over_th_limit_2 = 3
voltage_type_1 = f'W={window_size_2}; F={filter_size_2}; T={over_th_limit_2}'
# ## Sample rates:
sample_rate_pwr = 20.8   # should be 35
sample_rate_spi = 50e3
sample_rate_scope = 100e3
cut_scope_at = 790790


def main():
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    run_main()
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


def run_main():
    list_of_figs = []
    if kill_chrome:
        os.system("taskkill /im chrome.exe /f")
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(
        log_file.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    log_spi_files, log_spi_names, log_scope_files, log_scope_names = list(
        log_spi_or_scope.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    plot_titles = ['PWR Prints - DC Voltage [V]', 'SPI Prints - DC Voltage [V]', 'Scope Prints - DC Voltage [V]']
    plot_rows = int(len(plot_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))],
                           (plot_rows, plot_columns)).tolist()
    for index_file, file in enumerate(log_files_after_filter):
        if plots_per_test == 1:
            string_index_file = ''
        else:
            string_index_file = f'{log_file_names[index_file][:6]} - '
        print(f'Plotting record number {string_index_file}- {log_file_names[index_file]}:')

        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
         log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(file, test_type=test_type, extra_prints=8)
        plot_list = [log_voltage, log_spi_files[index_file]['Vdc'], log_scope_files[index_file]['V-inverter'][:cut_scope_at]]
        plot_frequency = [sample_rate_pwr, sample_rate_spi, sample_rate_scope]
        if index_file % plots_per_test == 0:
            fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns,
                                specs=all_specs, shared_xaxes=True)
        for index, plot in enumerate(plot_titles):
            fig.add_trace(go.Scatter(x=plot_list[index].index * (1 / plot_frequency[index]), y=plot_list[index],
                                     name=f"{string_index_file}{plot}"),
                          col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
            if index != 0:
                sample_rate_ratio = int(round(plot_frequency[index] / plot_frequency[0]))
                down_sampled_data = plot_list[index].iloc[::sample_rate_ratio].reset_index(drop=True)
                fig.add_trace(go.Scatter(x=down_sampled_data.index * (1 / plot_frequency[0]), y=down_sampled_data,
                                         name=f"{string_index_file}{plot} - Down sampled"),
                              col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
            if index == 2:
                final_plot = log_scope_files[index_file]['V-arc'][:cut_scope_at]
                fig.add_trace(go.Scatter(x=final_plot.index * (1 / plot_frequency[index]), y=final_plot,
                                         name='Scope Prints - Arc Voltage [V]'),
                              col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
        if index_file % plots_per_test == 0:
            list_of_figs.append(fig)
    if plot_offline:
        for index_fig, fig in enumerate(list_of_figs):
            fig.update_layout(title=plot_name, title_font_color="#407294", title_font_size=40,
                              legend_title="Records:", legend_title_font_color="green")
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, auto_open=auto_open_chrome_output,
                                filename=f'{path_output}/{log_file_names[index_fig][:32]} - plot_name.html')
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
