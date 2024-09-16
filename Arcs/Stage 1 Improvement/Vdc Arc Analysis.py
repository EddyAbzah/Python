import os
import gc
import sys
import inspect
from datetime import datetime
# import math
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import arc_th_calc
import log_file
import log_spi_or_scope
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # txt output instead of the console - ATTENTION - if True, there will be no Console output:
output_text = False
path_txt = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
path_txt = f'Terminal Log ({path_txt}).txt'
# # Chrome action for plotly:
kill_chrome = False
auto_open_chrome_output = True
# # Inverter stuff:
inverter_type = 'Jupiter_DSP'
log_with_time_stamp = True
# # Test type:
# # 1 = Regular Arc search (via State Machine).
# # 2 = False Alarms (no Arc search).
# # 3 = False Alarms, BUT, records will be cut manually when the state machine is = 10.
# # 4 = No Filter.
test_type = 4
# # Folders and filters:

path_logs = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Plus (AP1160C-PB-06-FW1 RevB)\Vdc algorithm\Vdc Arcs 01 - Digital record (23-11-2022)'
path_output = path_logs + '\\Graphs'
path_logs_string_filter = 'rec'
# # this will run the voltage algorithm on SPI records instead of the PWR
take_spi_voltage = True
spi_log_column = 'Vdc'
# ## Venus3 DSP: ## sample_rate_spi = 50e3
sample_rate_spi = 16667
sample_rate_pwr = 28.6
# # plot name ('.html' ass added later):
plot_name = f'Over Power Vdc Arcs'
# # divide the output plots into 2D figures:
plot_columns = 3
# # Energy Rise and Current Drop parameters:
window_size_1 = 20
filter_size_1 = 15
over_th_limit_1 = 12
# ## set sync_voltage_detection (in "arc_th_calc.py") to 9 if 5;5;3  _  or to 10 if 3;3;2
window_size_2 = 20
filter_size_2 = 15
over_th_limit_2 = 12
voltage_type_1 = f'W={window_size_2}; F={filter_size_2}; T={over_th_limit_2}'
# # Alpha / Beta Filter = smaller alpha, higher filtration:
vdc_alpha_filter = [0.0002, 0.0001, 0.00008]


def main():
    list_of_figs = []
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        default_stdout = sys.stdout
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        sys.stdout = open(f'{path_output}/{path_txt}', 'w')
    print(f"Starting Python... Time: {str(datetime.now())[:-7]}")
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(
        log_file.get_files(folder_path=path_logs, string_filter=path_logs_string_filter))
    plot_titles = ['Energy [dB]', 'AC Current [A]', f'DC Voltage [V]',
                   'Energy Rise [dB]', 'Current Drop [A]', 'Voltage Drop [V]']
    plot_rows = int(len(plot_titles) / plot_columns)
    all_specs = np.reshape([[{"secondary_y": False}] for x in range(len(plot_titles))], (plot_rows, plot_columns)).tolist()
    for index_file, file in enumerate(log_files_after_filter):
        print(f'Plotting record number {index_file} - {log_file_names[index_file]}:')

        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2, log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(file, test_type=test_type, extra_prints=8)
        energy_th_list = arc_th_calc.energy_rise_algo2(log_energy_before_stage2, window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        current_th_list = arc_th_calc.current_drop_algo2(log_current_before_stage2, window_size=window_size_1, filter_size=filter_size_1, over_th_limit=over_th_limit_1)
        list_log_voltage = []
        voltage_th_list_avg = []
        log_spi_files, log_spi_names, log_scope_files, log_scope_names = list(log_spi_or_scope.get_files(folder_path=path_logs, string_filter=path_logs_string_filter, spi_log_column=spi_log_column, file_name=log_file_names[index_file][-14:-4].replace("pwr", "spi")))
        for alpha_filter in vdc_alpha_filter:
            print(f'Vdc => Filtering the SPI')
            voltage_alpha_filtered = log_file.convert_to_df(log_file.alpha_beta_filter(log_spi_files[0], alpha_filter, stop_if_0=True))
            print(f'Vdc => Downsampling')
            list_log_voltage.append(log_file.convert_to_df(log_file.avg_no_overlap(voltage_alpha_filtered, sample_rate_spi, sample_rate_pwr)))
            print(f'Vdc => Calculating the TH')
            voltage_th_list_avg.append(arc_th_calc.voltage_drop_algo2(list_log_voltage[-1], window_size=window_size_2, filter_size=filter_size_2, over_th_limit=over_th_limit_2))

        plot_list = [log_energy, log_current, list_log_voltage, energy_th_list, current_th_list, voltage_th_list_avg]
        fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns, specs=all_specs, shared_xaxes=True)
        fig_counter = 0
        for index, plot in enumerate(plot_titles):
            if index != 2 and index != 5:
                fig.add_trace(go.Scatter(y=plot_list[index], name=plot), col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                fig_counter += 1
            else:
                if index == 2:
                    fig.add_trace(go.Scatter(y=log_voltage, name='PWR print DC Voltage [V]'), col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                    fig_counter += 1
                for index_2, plot_2 in enumerate(plot_list[index]):
                    fig.add_trace(go.Scatter(y=plot_2, name=f"{plot} - Alpha Filter = {vdc_alpha_filter[index_2]}", showlegend=True, visible=False), col=index % plot_columns + 1, row=int(index / plot_columns) + 1)
                fig.data[fig_counter + 1].visible = True
                fig_counter += len(plot_list[index])
            list_of_figs.append(fig)
        for index_fig, fig in enumerate(list_of_figs):
            slider_steps = []
            for index_alpha_filter, alpha_filter in enumerate(vdc_alpha_filter):
                step = dict(args=[{"visible": [True] * 3 + [False] * len(vdc_alpha_filter) + [True] * 2 + [False] * len(vdc_alpha_filter)}, ], label=alpha_filter)
                step["args"][0]["visible"][3 + index_alpha_filter] = True
                step["args"][0]["visible"][3 + len(vdc_alpha_filter) + 2 + index_alpha_filter] = True
                slider_steps.append(step)
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(prefix='Alpha Filter = ', xanchor="center", font=dict(size=16)))])
        fig.update_layout(title=log_file_names[index_file][:-4], title_font_color="#407294", title_font_size=40, legend_title="Records:", legend_title_font_color="green")
        if kill_chrome:
            os.system("taskkill /im chrome.exe /f")
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{log_file_names[index_file][:-4]}.html', auto_open=auto_open_chrome_output)
    print(f'Python finished... Time: {str(datetime.now())[:-7]}')
    if output_text and inspect.currentframe().f_code.co_name == 'main':
        sys.stdout.close()
        sys.stdout = default_stdout


if __name__ == "__main__":
    main()
