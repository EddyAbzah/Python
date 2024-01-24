from io import StringIO
import pandas as pd
import plotly
import plotly.graph_objects as go
# import pyinstaller as pyinstaller
from plotly.subplots import make_subplots
import numpy as np
import log_file
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


# # Folders and filters:
inverter_type = 'Venus3 Analog'
log_with_time_stamp = True
path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\MANA'
path_log_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\MANA'
file_number = 'Rec108'
file_name = f'{file_number} - Master PWR prints'
# file_name = f'{file_number} - Master and Slave PWR prints'
path_log_name_1 = f'FAs 13 {file_number} - Master pwr'
path_log_name_2 = f'FAs 13 {file_number} - Slave pwr'
add_zes = False
path_log_name_3 = f'FAs 13 {file_number} - Master ZES.log'
# # Chrome action for plotly:
plot_offline = True
kill_chrome = False and plot_offline
auto_open_chrome_output = True and plot_offline
plot_titles = ['Master - Energy', 'Slave - Energy', 'Master - Vdc', 'Slave - Vdc',
               'Master - Iac', 'Slave - Iac', 'Master - Machine State', 'Slave - Machine State']
if add_zes:
    plot_titles.extend('Master - ZES', 'Slave - ZES')
plot_columns = 2
plot_rows = 4
add_data_labels_1 = [754.687500, 1674, 'FA Arc here']
add_data_labels_2 = [0, 0, 'FA Arc in Master']
slave_only = False
master_only = True or slave_only


def main():
    global plot_titles
    log_file.choose_inverter(inverter_string=inverter_type, with_time_stamp=log_with_time_stamp)
    log_files_after_filter, log_file_names, log_arc_detections = list(log_file.get_files(
        folder_path=path_log_folder, string_filter=file_number, dropna=False))
    (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
     log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(log_files_after_filter[0],
                                                                                         test_type=4, extra_prints=8)
    df_1 = [log_energy, log_voltage, log_current, log_state]
    if add_data_labels_1[0] > 0:
        for i_v, v in enumerate(log_voltage):
            if v == add_data_labels_1[0]:
                if log_power[i_v] == add_data_labels_1[1]:
                    add_data_labels_1[1] = i_v
                    add_data_labels_2[1] = i_v
    if slave_only:
        for i in range(0, len(plot_titles), 2):
            temp = plot_titles[i]
            plot_titles[i] = plot_titles[i + 1]
            plot_titles[i + 1] = temp
    if not master_only:
        (log_energy, log_current, log_voltage, log_power, log_state, log_energy_before_stage2, log_current_before_stage2,
         log_voltage_before_stage2, log_power_before_stage2, cut_log_at) = log_file.get_logs(log_files_after_filter[1],
                                                                                             test_type=4, extra_prints=8)
        df_2 = [log_energy, log_voltage, log_current, log_state]
        if add_data_labels_2[0] > 0:
            for i_v, v in enumerate(log_voltage):
                if v == add_data_labels_2[0]:
                    if log_power[i_v] == add_data_labels_2[1]:
                        add_data_labels_2[1] = i_v
                        add_data_labels_1[1] = i_v
    if master_only:
        all_specs = np.reshape([[{"secondary_y": False}] for x in range(1 * plot_rows)], (plot_rows, 1)).tolist()
        fig = make_subplots(subplot_titles=plot_titles[::2], rows=plot_rows, cols=1, specs=all_specs, shared_xaxes=True)
    else:
        all_specs = np.reshape([[{"secondary_y": False}] for x in range(plot_columns * plot_rows)], (plot_rows, plot_columns)).tolist()
        fig = make_subplots(subplot_titles=plot_titles, rows=plot_rows, cols=plot_columns, specs=all_specs, shared_xaxes=True)
    for i in range(int(len(plot_titles) / 2)):
        fig.add_trace(go.Scatter(y=df_1[i], name=f'{path_log_name_1}: {plot_titles[i * 2]}', showlegend=True), col=1, row=i + 1)
        if not master_only:
            fig.add_trace(go.Scatter(y=df_2[i], name=f'{path_log_name_2}: {plot_titles[i * 2 + 1]}', showlegend=True), col=2, row=i + 1)
        if add_data_labels_1[1] != 0:
            fig.add_annotation(row=i + 1, col=1, x=add_data_labels_1[1], y=df_1[i][add_data_labels_1[1]],
                               text=add_data_labels_1[2], showarrow=True, opacity=1)
        if add_data_labels_2[1] != 0 and not master_only:
            fig.add_annotation(row=i + 1, col=2, x=add_data_labels_2[1], y=df_2[i][add_data_labels_1[1]],
                               text=add_data_labels_1[2], showarrow=True, opacity=1)
    if add_zes:
        fig.add_trace(go.Scatter(y=df_1[i], name=f'{path_log_name_1}: {plot_titles[i * 2]}', showlegend=True), col=1, row=i + 1)
        if not master_only:
            fig.add_trace(go.Scatter(y=df_2[i], name=f'{path_log_name_2}: {plot_titles[i * 2 + 1]}', showlegend=True), col=2, row=i + 1)
    print("Done")
    if plot_offline:
        fig.update_layout(title=file_name, title_font_color="#407294", title_font_size=40, legend_title="Prints:", legend_title_font_color="green")
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{file_name}.html', auto_open=auto_open_chrome_output)


if __name__ == "__main__":
    main()
