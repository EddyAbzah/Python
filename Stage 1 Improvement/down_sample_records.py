import os
import gc
import sys
import inspect
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt      # for plt.plot()
import numpy as np
import pandas as pd
import arc_th_calc
import log_spi_or_scope
import log_file
from matplotlib import pyplot
_P = pyplot.plot


# ## txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# ## Chrome action for plotly:
plot_offline = True
auto_open_chrome_output = plot_offline and False
kill_chrome = plot_offline and False
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
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Temp'
path_logs = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\Temp'
# ## Leave empty if you would like to skip:
path_logs_string_filter = 'FA Rec022'
# ## plot name ('.html' ass added later):
# ## divide the output plots into 2D figures:
plot_columns = 1
# ## Test parameters:
sample_rate_pwr = 20.8   # should be 35
sample_rate_interrupt = 50
sample_rate_new = 50e3
sample_rate_plot = 100
alpha_filters = np.arange(0.008, 0, -0.002)
alpha_filters = [0.02, 0.01, 0.005, 0.0001]
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size = 5
filter_size = 5
over_th_limit = 3
voltage_T_avg_F_min = False
voltage_drop_string = f'W={window_size}; F={filter_size}; T={over_th_limit}'
if voltage_T_avg_F_min:
    voltage_type_string = 'AVG'
else:
    voltage_type_string = 'MIN'
# ## Summary:
plot_name = f'Detection for Voltage algorithm {voltage_type_string} @ {voltage_drop_string}'
print_method_calls = True


# #######   IMPORTANT:   this script need both PWR and PSI files   ####### #


def main():
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    global window_size
    global filter_size
    global over_th_limit
    global voltage_T_avg_F_min
    global voltage_drop_string
    global voltage_type_string
    global plot_name
    for detection_algo in [[5, 5, 3], [10, 8, 5], [20, 15, 12]]:
        for T_avg_F_min in [True, False]:
            window_size = detection_algo[0]
            filter_size = detection_algo[1]
            over_th_limit = detection_algo[2]
            voltage_T_avg_F_min = T_avg_F_min
            if voltage_T_avg_F_min:
                voltage_type_string = 'AVG'
            else:
                voltage_type_string = 'MIN'
            voltage_drop_string = f'W={window_size}; F={filter_size}; T={over_th_limit}'
            run_main()
            gc.collect()
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


def run_main(mana, sex):
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
    plot_titles = ['PWR Prints [35Hz]', 'SPI Prints [50kHz]', 'SPI Prints - IIR filtered [50kHz]',
                   'SPI Prints - down sampled after IIR [35Hz]', f'Voltage Drop {voltage_type_string} ({voltage_drop_string})']
    trace_titles = ['PWR Prints - unmodified', 'SPI Prints - unmodified', 'SPI Prints - IIR filtered',
                    'SPI Prints - down sampled after IIR', f'Voltage Drop']
    plot_rows = int(len(plot_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))],
                           (plot_rows, plot_columns)).tolist()
    for index_file, file in enumerate(log_files_after_filter):
        gc.collect()
        alpha_filtered_logs = []
        down_sampled_logs = []
        voltage_drop_logs = []
        voltage_drop_strings = []
        if plots_per_test == 1:
            string_index_file = ''
        else:
            string_index_file = f'{log_file_names[index_file][:6]} - '
        print(f'Plotting record number {string_index_file}- {log_file_names[index_file]}:')

        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
         log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(file, test_type=test_type, extra_prints=8)
        plot_list = [log_voltage] + [log_spi_files[index_file]['Vdc']] * 4
        plot_frequency = [sample_rate_pwr] + [sample_rate_new] * 2 + [sample_rate_pwr] * 2
        if index_file % plots_per_test == 0:
            fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns,
                                specs=all_specs, shared_xaxes=True)
        if print_method_calls:
            all_plots_count = 0
        for index, plot in enumerate(plot_list):
            if index == 0:
                fig.add_trace(go.Scattergl(x=plot.index * (1 / plot_frequency[index]), y=plot,
                                           name=f"{string_index_file}{trace_titles[index]}"),
                              col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                if print_method_calls:
                    all_plots_count = all_plots_count + 1
                    print(f'plot index = {index} → all_plots_count = {all_plots_count}')
            elif index == 1:
                down_sampled_df = log_file.avg_no_overlap(plot, plot_frequency[index], sample_rate_plot)
                fig.add_trace(go.Scattergl(x=plot.index * (1 / sample_rate_plot), y=down_sampled_df,
                                           name=f"{string_index_file}{trace_titles[index]}"),
                              col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                if print_method_calls:
                    all_plots_count = all_plots_count + 1
                    print(f'plot index = {index} → all_plots_count = {all_plots_count}')
            elif index == 2:
                for alpha in alpha_filters:
                    df_alpha_filtered = log_file.convert_to_df(log_file.alpha_beta_filter(plot, alpha))
                    down_sampled_df = log_file.avg_no_overlap(df_alpha_filtered, plot_frequency[index], sample_rate_plot)
                    fig.add_trace(go.Scattergl(x=df_alpha_filtered.index * (1 / sample_rate_plot), y=down_sampled_df,
                                               name=f"{string_index_file}{trace_titles[index]} (alpha = {alpha:.4f})",
                                               showlegend=True, visible=False),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    alpha_filtered_logs.append(df_alpha_filtered)
                    if print_method_calls:
                        all_plots_count = all_plots_count + 1
                        print(f'plot index = {index} → all_plots_count = {all_plots_count}')
                fig.data[2].visible = True
            elif index == 3:
                for index_df, df_alpha_filtered in enumerate(alpha_filtered_logs):
                    df_down_sampled = log_file.avg_no_overlap(df_alpha_filtered, sample_rate_new, sample_rate_pwr)
                    fig.add_trace(go.Scattergl(x=df_down_sampled.index * (1 / plot_frequency[index]), y=df_down_sampled,
                                               name=f"{string_index_file}{trace_titles[index]} (alpha = {alpha_filters[index_df]:.4f})",
                                               showlegend=True, visible=False),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    if print_method_calls:
                        all_plots_count = all_plots_count + 1
                        print(f'plot index = {index} → all_plots_count = {all_plots_count}')
                    down_sampled_logs.append(df_down_sampled)
                fig.data[2 + len(alpha_filters)].visible = True
            else:
                voltage_drop_logs.append(log_file.convert_to_df(arc_th_calc.voltage_algo(log_voltage, window_size=window_size,
                                                                                         filter_size=filter_size, over_th_limit=over_th_limit,
                                                                                         true_if_avg=voltage_T_avg_F_min)))
                voltage_drop_strings.append(' - original')
                for alpha_index, alpha in enumerate(alpha_filters):
                    voltage_drop_logs.append(log_file.convert_to_df(arc_th_calc.voltage_algo(down_sampled_logs[alpha_index],
                                                                                             window_size=window_size,
                                                                                             filter_size=filter_size,
                                                                                             over_th_limit=over_th_limit,
                                                                                             true_if_avg=voltage_T_avg_F_min)))
                    voltage_drop_strings.append(f' - alpha = {alpha:.4f}')
                for df_index, df_voltage_drop in enumerate(voltage_drop_logs):
                    fig.add_trace(go.Scattergl(x=df_voltage_drop.index * (1 / plot_frequency[index]), y=df_voltage_drop,
                                               name=f"{string_index_file}{trace_titles[index]}{voltage_drop_strings[df_index]}",
                                               showlegend=True, visible=False),
                                  col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    if print_method_calls:
                        all_plots_count = all_plots_count + 1
                        print(f'plot index = {index} → all_plots_count = {all_plots_count}')
                fig.data[2 + len(alpha_filters) * 2].visible = True
                fig.data[3 + len(alpha_filters) * 2].visible = True
        if index_file % plots_per_test == 0:
            list_of_figs.append(fig)
    if plot_offline:
        slider_steps = []
        for alpha_index, alpha in enumerate(alpha_filters):
            step = dict(args=[{"visible": [True] * 2 + [False] * len(alpha_filters) * 2 + [True] + [False] * len(alpha_filters)},
                              ], label=f'Alpha of {alpha:.4f}')
            step["args"][0]["visible"][2 + alpha_index] = True
            step["args"][0]["visible"][2 + len(alpha_filters) + alpha_index] = True
            step["args"][0]["visible"][3 + len(alpha_filters) * 2 + alpha_index] = True
            slider_steps.append(step)
        for index_fig, fig in enumerate(list_of_figs):
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200",
                                            currentvalue=dict(prefix='SPI Prints - IIR filter with ',
                                                              xanchor="center", font=dict(size=16)))])
            fig.update_layout(title=plot_name, title_font_color="#407294", title_font_size=40,
                              legend_title="Records:", legend_title_font_color="green")
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, auto_open=auto_open_chrome_output,
                                filename=f'{path_output}/{log_file_names[index_fig]} - {plot_name}.html')
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
