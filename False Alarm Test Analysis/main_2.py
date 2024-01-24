import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


class F_Event:
    """This is the event for stage 1, error code 18xF"""

    def __init__(self, date, time, code, Arc_power_diff, Bitmap, Max_Phase_shift, Amp_change_diff,
                 Amp_change_percentage, full_time):
        self.date = date
        self.time = time
        self.code = code
        self.Arc_power_diff = Arc_power_diff
        if Arc_power_diff > 2500:
            print("nanana")
        self.Bitmap = BITMAP(Bitmap)
        self.Max_Phase_shift = Max_Phase_shift
        self.Amp_change_diff = Amp_change_diff
        self.Amp_change_percentage = Amp_change_percentage
        self.Date_hour = full_time
        if Bitmap >= 4:
            self.pass_stage_2 = 1
        else:
            self.pass_stage_2 = 0


class D_Event:
    """This is the event for stage 2, error code 18xD"""

    def __init__(self, date, time, code, Mngr_pwr, ArcMaxEnergy, ArcMaxCurrentDrop, ArcAverageIac, ArcminEnergy,
                 full_time):
        self.date = date
        self.time = time
        self.code = code
        self.Mngr_pwr = Mngr_pwr

        self.ArcMaxEnergy = ArcMaxEnergy
        self.ArcMaxCurrentDrop = ArcMaxCurrentDrop
        self.ArcAverageIac = ArcAverageIac
        self.ArcminEnergy = ArcminEnergy
        self.Date_hour = full_time


class BITMAP:
    def __init__(self, Bitmap):
        print((Bitmap))
        if math.isnan(Bitmap):
            print("on it")
            Bitmap = 0;
        if Bitmap > 2 ** 6:
            print("on it")
            Bitmap = 0;
        if Bitmap < 0:
            print("on it")
            Bitmap = 0;
        self.bitmap_decimal = int(Bitmap)

        self.bitmap_binary = bin(int(Bitmap))[2:]
        temp = bin(int(Bitmap))[2:]
        mantisa = [0, 0, 0, 0, 0, 0]
        for i in range(len(temp)):
            mantisa[i] = temp[i]
        mantisa = mantisa[::-1]
        self.power_diff_bit = int(mantisa[5])
        obs = mantisa[4]
        self.phase_shift_diff = int(mantisa[3])
        self.amplitude_bit = int(mantisa[2])
        self.overpower_bit = int(mantisa[1])
        self.unsynced_bit = int(mantisa[0])

        print("hey")


def sort_by_values_len(dict):
    dict_len = {key: len(value) for key, value in dict.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = [{item[0]: dict[item[0]]} for item in sorted_key_list]
    return sorted_dict


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def plot_4_pane(subplot_title1, subplot_title2, subplot_title3, subplot_title4, series, key, Arc_power_diff,
                Max_Phase_shift, Amp_change_percentage,
                Amp_change_diff, Arc_power_diff_th_list, MaxPhaseshift_th_list,
                Amp_change_percentage_th_list, Amp_change_diff_th_list, time, bitmap_list, results_path):
    plot_titles = [subplot_title1, subplot_title2, subplot_title3, subplot_title4]
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 4
    col = 1
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig_18xF = make_subplots(rows=row, cols=col,
                             specs=all_specs_reshaped, shared_xaxes=True,
                             subplot_titles=plot_titles)

    power_diff_bit = [x.power_diff_bit for x in bitmap_list]
    Max_Phase_shift_bit = [x.phase_shift_diff for x in bitmap_list]
    amplitude_bit = [x.amplitude_bit for x in bitmap_list]
    overpower_bit = [x.overpower_bit for x in bitmap_list]
    unsynced_bit = [x.unsynced_bit for x in bitmap_list]
    bitmap_decimal_bit = [x.bitmap_decimal for x in bitmap_list]

    customdata = [power_diff_bit, Max_Phase_shift_bit, amplitude_bit, overpower_bit, unsynced_bit]
    marker_Phase_shift_dict = {'size': [], 'color': [], 'symbol': []}
    marker_power_dict = {'size': [], 'color': [], 'symbol': []}
    marker_amplitude_dict = {'size': [], 'color': [], 'symbol': []}
    # marker=dict(size=[40, 60, 80, 100],
    #                 color=[0, 1, 2, 3])

    for i in range(len(Max_Phase_shift)):

        if Max_Phase_shift[i] > MaxPhaseshift_th_list[0] and Max_Phase_shift_bit[i] > 0:

            marker_Phase_shift_dict['size'].append(15)
            marker_Phase_shift_dict['color'].append("#ff0000")
            marker_Phase_shift_dict['symbol'].append(2)
        else:
            marker_Phase_shift_dict['size'].append(7)
            marker_Phase_shift_dict['color'].append("#f0a000")
            marker_Phase_shift_dict['symbol'].append(0)

    for i in range(len(Arc_power_diff)):

        if Arc_power_diff[i] > Arc_power_diff_th_list[0] and power_diff_bit[i] > 0:

            marker_power_dict['size'].append(15)
            marker_power_dict['color'].append("#ff0000")
            marker_power_dict['symbol'].append(2)

        else:
            marker_power_dict['size'].append(7)
            marker_power_dict['color'].append("#00a0f0")
            marker_power_dict['symbol'].append(0)

    for i in range(len(Amp_change_diff)):

        if Amp_change_diff[i] > Amp_change_diff_th_list[0] and amplitude_bit[i] > 0:

            marker_amplitude_dict['size'].append(15)
            marker_amplitude_dict['color'].append("#ff0000")
            marker_amplitude_dict['symbol'].append(2)

        else:
            marker_amplitude_dict['size'].append(7)
            marker_amplitude_dict['color'].append("#21ad3d")
            marker_amplitude_dict['symbol'].append(0)

    fig_18xF.add_trace(
        go.Scattergl(y=Arc_power_diff, x=time,
                     name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title1,
                     mode="markers", marker=marker_power_dict, customdata=list(zip(bitmap_decimal_bit, power_diff_bit)),
                     hovertemplate="<br>".join(
                         ["Time: %{x}", "value: %{y}", "decimal bitmap;power diff bit: %{customdata}"])
                     ), row=1, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=Max_Phase_shift, x=time,
                     name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title2,
                     mode="markers", marker=marker_Phase_shift_dict,
                     customdata=list(zip(bitmap_decimal_bit, Max_Phase_shift_bit)),
                     hovertemplate="<br>".join(
                         ["Time: %{x}", "value: %{y}", "decimal bitmap;Max Phase shiftbit: %{customdata}"])
                     ),
        row=2, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=Amp_change_percentage, x=time,
                     name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title3,
                     mode="markers", marker=marker_amplitude_dict, customdata=customdata),
        row=3, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=Amp_change_diff, x=time, name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title4,
                     mode="markers", customdata=list(zip(bitmap_decimal_bit, amplitude_bit)),
                     hovertemplate="<br>".join(
                         ["Time: %{x}", "value: %{y}", "decimal bitmap;amplitude_bit: %{customdata}"])),
        row=4, col=1)

    "plot Th"
    fig_18xF.add_trace(
        go.Scattergl(y=Arc_power_diff_th_list, x=time, name=key + " " + subplot_title1 + ' th'),
        row=1, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=MaxPhaseshift_th_list, x=time, name=key + " " + subplot_title2 + ' th'),
        row=2, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=Amp_change_percentage_th_list, x=time, name=key + " " + subplot_title3 + ' th'),
        row=3, col=1)
    fig_18xF.add_trace(
        go.Scattergl(y=Amp_change_diff_th_list, x=time, name=key + " " + subplot_title4 + ' th'),
        row=4, col=1)
    config = {'scrollZoom': True}

    """Arc Detected = (Power Diff)   AND   ((Phase Shift) OR (Amp Change))"""

    fig_18xF.update_layout(
        title_font_family="Times New Roman",
        title_font_color="red",
        title={
            'text': 'Params of stage 1  for SN: ' + key + "<br>"  "Arc Detected = (Power Diff)   AND   ((Phase Shift) OR (Amp Change))",
            'font': dict(
                family="Arial",
                size=20,
                color='blue')
        }
    )

    # fig_18xF.update_layout(
    #     xaxis4=dict(
    #         rangeselector=dict(
    #             buttons=list([
    #                 dict(count=1,
    #                      step="all",
    #                      stepmode="backward"),
    #             ])
    #         ),
    #         rangeslider=dict(
    #             visible=True
    #         ),
    #     )
    # )###
    fig_18xF.update_layout(
        xaxis4=dict(
            autorange=False,
            range=[time[0], time[-1]],
            rangeselector=dict(
                buttons=list([
                    dict(count=100,
                         label="minute",
                         step="minute", stepmode="todate",
                         ),
                    dict(count=100,
                         label="hour",
                         step="hour", stepmode="todate"),
                    dict(count=100,
                         label="day",
                         step="day",
                         stepmode="todate"),
                    dict(count=1100,
                         label="month",
                         step="month",
                         stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True,
                autorange=True,
                range=[time[0], time[-1]]
            ),
            type="date"

        )
    )
    ###"""

    # fig_18xF.update_layout(
    #     xaxis4=dict(
    #         autorange=True,
    #         range=[time[0], time[-1]],
    #         rangeslider=dict(
    #             autorange=True,
    #             range=[time[0], time[-1]]
    #         ),
    #         type="date"
    #     ))
    fig_18xF.update_layout(xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.1)
    plotly.offline.plot(fig_18xF, config=config, filename=results_path + '\\' + key + ' Arc params ' + str(
        round(series.loc[key], 2)) + "%" + date + '.html')


def plot_4_histogram(subplot_title1, subplot_title2, subplot_title3, subplot_title4, series, key, Arc_power_diff,
                     Max_Phase_shift, Amp_change_percentage, Amp_change_diff,
                     Arc_power_diff_th_list, MaxPhaseshift_th_list,
                     Amp_change_percentage_th_list, Amp_change_diff_th_list, time, bitmap_list, results_path):
    bigger_then_th_power_diff = round(
        len([item for item in Arc_power_diff if item > Arc_power_diff_th_list[0]]) / len(Arc_power_diff) * 100, 2)
    bigger_then_th_Max_Phase_shift = round(
        len([item for item in Max_Phase_shift if item > MaxPhaseshift_th_list[0]]) / len(Max_Phase_shift) * 100, 2)
    bigger_then_th_Amp_change_percentage = round(
        len([item for item in Amp_change_percentage if item > Amp_change_percentage_th_list[0]]) / len(
            Amp_change_percentage) * 100, 2)
    bigger_then_th_Amp_change_diff = round(
        len([item for item in Amp_change_diff if item > Amp_change_diff_th_list[0]]) / len(Amp_change_diff) * 100, 2)

    plot_titles = [subplot_title1 + "<br>" + str(bigger_then_th_power_diff) + "% passed Th",
                   subplot_title2 + "<br>" + str(bigger_then_th_Max_Phase_shift) + "% passed Th",
                   subplot_title3 + "<br>" + str(bigger_then_th_Amp_change_percentage) + "% passed Th",
                   subplot_title4 + "<br>" + str(bigger_then_th_Amp_change_diff) + "% passed Th"]
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 2
    col = 2
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=False,
                        subplot_titles=plot_titles,
                        )
    # title = 'Stage 1 histograms for SN: ' + key

    power_diff_bit = [x.power_diff_bit for x in bitmap_list]
    Max_Phase_shift_bit = [x.phase_shift_diff for x in bitmap_list]
    amplitude_bit = [x.amplitude_bit for x in bitmap_list]
    overpower_bit = [x.overpower_bit for x in bitmap_list]
    unsynced_bit = [x.unsynced_bit for x in bitmap_list]

    fig.add_trace(
        go.Histogram(x=Arc_power_diff, nbinsx=500, name=subplot_title1),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=Arc_power_diff_th_list[:10], nbinsx=500, name=subplot_title1 + " Th"),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=Max_Phase_shift, nbinsx=500, name=subplot_title2),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=MaxPhaseshift_th_list[:10], nbinsx=500, name=subplot_title2 + " Th"),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=Amp_change_percentage, nbinsx=500, name=subplot_title3),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=Amp_change_percentage_th_list[:10], nbinsx=500, name=subplot_title3 + " Th"),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=Amp_change_diff, nbinsx=500, name=subplot_title4),
        row=2, col=2)
    fig.add_trace(
        go.Histogram(x=Amp_change_diff_th_list[:10], nbinsx=500, name=subplot_title4 + " Th"),
        row=2, col=2)

    # fig.updtae
    config = {'scrollZoom': True}

    fig.update_layout(
        title={
            'text': 'Stage 1 histograms for SN: ' + key,
            'font': dict(
                family="Arial",
                size=20,
                color='blue')
        }
    )

    plotly.offline.plot(fig, config=config, filename=results_path + '\\' + key + ' Arc histogram Stage1, ' + str(
        round(series.loc[key], 2)) + "%" + date + '.html')
    print("hey")


def plot_4_pane_D(subplot_title1, subplot_title2, subplot_title3, subplot_title4, series, key, Arc_power_diff,
                  Max_Phase_shift, Amp_change_percentage,
                  Amp_change_diff, Arc_power_diff_th_list, MaxPhaseshift_th_list,
                  Amp_change_percentage_th_list, Amp_change_diff_th_list, time, bitmap_list, results_path):
    plot_titles = [subplot_title1, subplot_title2, subplot_title3, subplot_title4]
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 4
    col = 1
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig_18xF = make_subplots(rows=row, cols=col,
                             specs=all_specs_reshaped, shared_xaxes=True,
                             subplot_titles=plot_titles)

    power_diff_bit = [x.power_diff_bit for x in bitmap_list]
    Max_Phase_shift_bit = [x.phase_shift_diff for x in bitmap_list]
    amplitude_bit = [x.amplitude_bit for x in bitmap_list]
    overpower_bit = [x.overpower_bit for x in bitmap_list]
    unsynced_bit = [x.unsynced_bit for x in bitmap_list]
    bitmap_decimal_bit = [x.bitmap_decimal for x in bitmap_list]

    customdata = [power_diff_bit, Max_Phase_shift_bit, amplitude_bit, overpower_bit, unsynced_bit]
    marker_Phase_shift_dict = {'size': [], 'color': [], 'symbol': []}
    marker_power_dict = {'size': [], 'color': [], 'symbol': []}
    marker_amplitude_dict = {'size': [], 'color': [], 'symbol': []}
    # marker=dict(size=[40, 60, 80, 100],
    #                 color=[0, 1, 2, 3])

    for i in range(len(Max_Phase_shift)):

        if Max_Phase_shift[i] > MaxPhaseshift_th_list[0] and Max_Phase_shift_bit[i] > 0:

            marker_Phase_shift_dict['size'].append(15)
            marker_Phase_shift_dict['color'].append("#ff0000")
            marker_Phase_shift_dict['symbol'].append(2)
        else:
            marker_Phase_shift_dict['size'].append(7)
            marker_Phase_shift_dict['color'].append("#f0a000")
            marker_Phase_shift_dict['symbol'].append(0)

    for i in range(len(Arc_power_diff)):

        if Arc_power_diff[i] > Arc_power_diff_th_list[0] and power_diff_bit[i] > 0:

            marker_power_dict['size'].append(15)
            marker_power_dict['color'].append("#ff0000")
            marker_power_dict['symbol'].append(2)

        else:
            marker_power_dict['size'].append(7)
            marker_power_dict['color'].append("#00a0f0")
            marker_power_dict['symbol'].append(0)

    for i in range(len(Amp_change_diff)):

        if Amp_change_diff[i] > Amp_change_diff_th_list[0] and amplitude_bit[i] > 0:

            marker_amplitude_dict['size'].append(15)
            marker_amplitude_dict['color'].append("#ff0000")
            marker_amplitude_dict['symbol'].append(2)

        else:
            marker_amplitude_dict['size'].append(7)
            marker_amplitude_dict['color'].append("#21ad3d")
            marker_amplitude_dict['symbol'].append(0)

    fig_18xF.add_trace(
        go.Scatter(y=Arc_power_diff, x=time,
                   name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title1,
                   mode="markers", marker=marker_power_dict, customdata=list(zip(bitmap_decimal_bit, power_diff_bit)),
                   hovertemplate="<br>".join(
                       ["Time: %{x}", "value: %{y}", "decimal bitmap;power diff bit: %{customdata}"])
                   ),
        row=1, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=Max_Phase_shift, x=time,
                   name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title2,
                   mode="markers", marker=marker_Phase_shift_dict,
                   customdata=list(zip(bitmap_decimal_bit, Max_Phase_shift_bit)),
                   hovertemplate="<br>".join(
                       ["Time: %{x}", "value: %{y}", "decimal bitmap;Max Phase shiftbit: %{customdata}"])
                   ),
        row=2, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=Amp_change_percentage, x=time,
                   name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title3,
                   mode="markers", marker=marker_amplitude_dict, customdata=customdata),
        row=3, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=Amp_change_diff, x=time,
                   name=str(round(series.loc[key], 2)) + "% " + key + " " + subplot_title4,
                   mode="markers", customdata=list(zip(bitmap_decimal_bit, amplitude_bit)),
                   hovertemplate="<br>".join(
                       ["Time: %{x}", "value: %{y}", "decimal bitmap;amplitude_bit: %{customdata}"])),
        row=4, col=1)

    "plot Th"
    fig_18xF.add_trace(
        go.Scatter(y=Arc_power_diff_th_list, x=time, name=key + " " + subplot_title1 + ' th'),
        row=1, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=MaxPhaseshift_th_list, x=time, name=key + " " + subplot_title2 + ' th'),
        row=2, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=Amp_change_percentage_th_list, x=time, name=key + " " + subplot_title3 + ' th'),
        row=3, col=1)
    fig_18xF.add_trace(
        go.Scatter(y=Amp_change_diff_th_list, x=time, name=key + " " + subplot_title4 + ' th'),
        row=4, col=1)
    config = {'scrollZoom': True}

    """Arc Detected = (Power Diff)   AND   ((Phase Shift) OR (Amp Change))"""
    fig_18xF.update_layout(
        title_font_family="Times New Roman",
        title_font_color="red",
        title="Arc Detected = (Power Diff)   AND   ((Phase Shift) OR (Amp Change))")

    plotly.offline.plot(fig_18xF, config=config, filename=results_path + '\\' + key + ' Arc params ' + str(
        round(series.loc[key], 2)) + "%" + date + '.html')


def plot_4_histogram_D(subplot_title1, subplot_title2, subplot_title3, subplot_title4, series, key,
                       ArcMaxEnergy, ArcMaxCurrentDrop, ArcAverageIac, ArcminEnergy, time, numerofstage2, results_path):
    plot_titles = [subplot_title1,
                   subplot_title2,
                   subplot_title3,
                   subplot_title4, 'MaxEnergy-minEnergy', 'AverageIac-MaxCurrentDrop']
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 2
    col = 3
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=False,
                        subplot_titles=plot_titles
                        )

    fig.add_trace(
        go.Histogram(x=ArcMaxEnergy, nbinsx=100, name=subplot_title1),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=ArcMaxCurrentDrop, nbinsx=100, name=subplot_title2),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=ArcAverageIac, nbinsx=100, name=subplot_title3),
        row=1, col=3)

    fig.add_trace(
        go.Histogram(x=ArcminEnergy, nbinsx=100, name=subplot_title4),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=np.array(ArcMaxEnergy) - np.array(ArcminEnergy), nbinsx=100, name=plot_titles[4]),
        row=2, col=2)
    fig.add_trace(
        go.Histogram(x=np.array(ArcAverageIac) - np.array(ArcMaxCurrentDrop), nbinsx=100, name=plot_titles[5]),
        row=2, col=3)

    fig.update_layout(
        title={
            'text': 'Stage 2 histograms for SN: ' + key + ', ' + str(numerofstage2) + ' events',
            'font': dict(
                family="Arial",
                size=20,
                color='red')
        }
    )
    # fig.updtae
    config = {'scrollZoom': True}

    plotly.offline.plot(fig, config=config, filename=results_path + '\\' + key + ' Arc histogram Stage2, ' + str(
        round(series.loc[key], 2)) + "%," + date + '.html')
    print("hey")


def plot_avg_events_per_day(count_df, count_df_stage2, results_path, date, Stage_1_Real_Events, Stage_2_Real_Events):
    "Avg histograms per day"
    plot_titles = ["Average Stage 1 events per day", "Real Stage 1 events per day", "Average Stage 2 events per day",
                   "Real Stage 2 events per day"]
    all_specs = np.array([[{"secondary_y": True}] for x in range(2)])
    row = 1
    col = 2
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig_histogram_per_day = make_subplots(rows=row, cols=col,
                                          specs=all_specs_reshaped, shared_xaxes=False,
                                          )

    fig_histogram_per_day.add_trace(
        go.Histogram(x=count_df['Per day'], name="stage 1 avg ", nbinsx=500),
        row=1, col=1)
    fig_histogram_per_day.add_trace(
        go.Histogram(x=Stage_1_Real_Events, name="stage 1 real ", nbinsx=500),
        row=1, col=1)
    fig_histogram_per_day.add_trace(
        go.Histogram(x=count_df_stage2['Per day'], name="stage 2 avg ", nbinsx=500),
        row=1, col=2)
    fig_histogram_per_day.add_trace(
        go.Histogram(x=Stage_2_Real_Events, name="stage 2 real ", nbinsx=500),
        row=1, col=2)
    config = {'scrollZoom': True}
    fig_histogram_per_day.update_layout(
        title={
            'text': 'Events per day',
            'font': dict(
                family="Arial",
                size=20,
                color='green')
        }
    )
    plotly.offline.plot(fig_histogram_per_day, config=config,
                        filename=results_path + '\\' + 'Events per day(Real+Avg)' + date + '.html')


def summary_pie_plot(count_df, count_df_stage2, results_path, date):
    plot_titles = ["18xF-Stage 1", "Stage 1 Pie", "18xD && 18X100-Stage 2", "Stage 2 Pie"]
    all_specs = np.array([[{"type": "pie"}] for x in range(len(plot_titles))])
    row = 2
    col = 2
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig_pie = make_subplots(rows=row, cols=col,
                            specs=all_specs_reshaped, shared_xaxes=False,
                            subplot_titles=plot_titles)

    fig_pie.add_trace(go.Pie(
        values=count_df['count_stage_1'],
        labels=count_df['SN'],
        name="Pass stage 1"),
        row=1, col=2, )
    fig_pie.add_trace(go.Pie(
        values=count_df_stage2['count_stage_2'],
        labels=count_df_stage2['SN'],
        name="Pass stage 1"),
        row=2, col=2)

    fig_pie.update_traces(textposition='inside')

    fig_pie.add_trace(go.Table(header=dict(
        values=['Total inverters', 'Updated to AFCI2.2', 'Number of inverters' + '\n' + ' that pass stage 1',
                'number of ' + '\n' + 'stage 1 events']),
                               cells=dict(values=[number_of_invertes_in_fleet, number_of_UPDATED_AFCI,
                                                  total_pass_stage1_inverters, [sum_of_pass_stage1_events]])),
                      row=1, col=1)
    fig_pie.add_trace(go.Table(header=dict(
        values=['Total inverters', 'Updated to AFCI2.2', 'Number of inverters' + '\n' + ' that pass stage 2',
                'number of ' + '\n' + 'stage 2 events']),
                               cells=dict(values=[number_of_invertes_in_fleet, number_of_UPDATED_AFCI,
                                                  total_pass_stage2_inverters, [sum_of_pass_stage2_events]])),
                      row=2, col=1)
    config = {'scrollZoom': True}
    fig_pie.update_layout(showlegend=False, height=1000, width=1500)
    fig_pie.update_layout(
        title={
            'text': 'Stage 1 and Stage 2 table summary',
            'font': dict(
                family="Arial",
                size=20,
                color='green')
        }
    )
    plotly.offline.plot(fig_pie, config=config, filename=results_path + '\\' + 'Pie chart' + date + '.html')


def All_data_stage_1_histograms(df_18xF, Arc_power_diff_th, MaxPhaseshift_th, Amp_change_percentage_th,
                                Amp_change_diff_th, bitmap_list, results_path, date):
    bigger_then_th_power_diff = round(
        len([item for item in df_18xF['Arc_power_diff'] if item > Arc_power_diff_th]) / len(
            df_18xF['Arc_power_diff']) * 100, 2)
    bigger_then_th_Max_Phase_shift = round(
        len([item for item in df_18xF['Max_Phase_shift'] if item > MaxPhaseshift_th]) / len(
            df_18xF['Max_Phase_shift']) * 100, 2)
    bigger_then_th_Amp_change_percentage = round(
        len([item for item in df_18xF['Amp_change_percentage'] if item > Amp_change_percentage_th]) / len(
            df_18xF['Amp_change_percentage']) * 100, 2)
    bigger_then_th_Amp_change_diff = round(
        len([item for item in df_18xF['Amp_change_diff'] if item > Amp_change_diff_th]) / len(
            df_18xF['Amp_change_diff']) * 100, 2)

    titles = ["Arc power diff", "Max_Phase_shift", "Amp_change_percentage", "Amp_change_diff"]
    plot_titles = [titles[0] + "<br>" + str(bigger_then_th_power_diff) + "% passed Th",
                   titles[1] + "<br>" + str(bigger_then_th_Max_Phase_shift) + "% passed Th",
                   titles[2] + "<br>" + str(bigger_then_th_Amp_change_percentage) + "% passed Th",
                   titles[3] + "<br>" + str(bigger_then_th_Amp_change_diff) + "% passed Th"]
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 2
    col = 2
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=False,
                        subplot_titles=plot_titles)

    power_diff_bit = [x.power_diff_bit for x in bitmap_list]
    Max_Phase_shift_bit = [x.phase_shift_diff for x in bitmap_list]
    amplitude_bit = [x.amplitude_bit for x in bitmap_list]
    overpower_bit = [x.overpower_bit for x in bitmap_list]
    unsynced_bit = [x.unsynced_bit for x in bitmap_list]

    fig.add_trace(
        go.Histogram(x=df_18xF['Arc_power_diff'], nbinsx=1000, name='Arc power diff'),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=[Arc_power_diff_th] * 1000, nbinsx=1000, name='Arc power diff' + " TH"),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=df_18xF['Max_Phase_shift'], nbinsx=1000, name="Max_Phase_shift"),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=[MaxPhaseshift_th] * 1000, nbinsx=1000, name="Max_Phase_shift" + " TH"),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=df_18xF['Amp_change_percentage'], nbinsx=1000, name="Amp_change_percentage"),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=[Amp_change_percentage_th] * 1000, nbinsx=1000, name="Amp_change_percentage" + " TH"),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=df_18xF['Amp_change_diff'], nbinsx=1000, name="Amp_change_diff"),
        row=2, col=2)
    fig.add_trace(
        go.Histogram(x=[Amp_change_diff_th] * 1000, nbinsx=1000, name="Amp_change_diff" + " TH"),
        row=2, col=2)

    config = {'scrollZoom': True}
    fig.update_layout(
        title={
            'text': 'Stage 1 Histograms for All SN, Total ' + str(total_pass_stage1_inverters) + ' events',
            'font': dict(
                family="Arial",
                size=20,
                color='green')
        }
    )

    plotly.offline.plot(fig, config=config,
                        filename=results_path + '\\' + 'All data stage 1 histograms ' + date + '.html')


def All_data_stage_2_histograms(df_18xD, results_path, date):
    plot_titles = ["ArcMaxEnergy",
                   "ArcMaxCurrentDrop",
                   "ArcAverageIac",
                   "ArcminEnergy", 'MaxEnergy-minEnergy', 'AverageIac-MaxCurrentDrop']
    all_specs = np.array([[{"secondary_y": True}] for x in range(len(plot_titles))])
    row = 2
    col = 3
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=False,
                        subplot_titles=plot_titles
                        )

    fig.add_trace(
        go.Histogram(x=df_18xD['ArcMaxEnergy'], nbinsx=100, name=plot_titles[0]),
        row=1, col=1)
    fig.add_trace(
        go.Histogram(x=df_18xD['ArcMaxCurrentDrop'], nbinsx=100, name=plot_titles[1]),
        row=1, col=2)
    fig.add_trace(
        go.Histogram(x=df_18xD['ArcAverageIac'], nbinsx=100, name=plot_titles[2]),
        row=1, col=3)

    fig.add_trace(
        go.Histogram(x=df_18xD['ArcminEnergy'], nbinsx=100, name=plot_titles[3]),
        row=2, col=1)
    fig.add_trace(
        go.Histogram(x=df_18xD['ArcMaxEnergy'] - df_18xD['ArcminEnergy'], nbinsx=100, name=plot_titles[4]),
        row=2, col=2)
    fig.add_trace(
        go.Histogram(x=df_18xD['ArcAverageIac'] - df_18xD['ArcMaxCurrentDrop'], nbinsx=100, name=plot_titles[5]),
        row=2, col=3)

    config = {'scrollZoom': True}
    fig.update_layout(
        title={
            'text': 'Stage 2 Histograms for All SN, Total ' + str(total_pass_stage2_inverters) + ' events',
            'font': dict(
                family="Arial",
                size=20,
                color='green')
        }
    )
    plotly.offline.plot(fig, config=config,
                        filename=results_path + '\\' + 'All data stage 2 histograms ' + date + '.html')


def bulid_count(df_18xF, df_18xD):
    count_df = pd.DataFrame()
    count_df_stage2 = pd.DataFrame()
    "--------------"
    count_df['count_stage_1'] = df_18xF['SN'].value_counts()
    count_df_stage2['count_stage_2'] = df_18xD['SN'].value_counts()
    "--------------"
    count_df['SN'] = count_df.index
    count_df_stage2['SN'] = count_df_stage2.index
    "--------------"
    sum_of_pass_stage1_events = count_df['count_stage_1'].sum()
    sum_of_pass_stage2_events = count_df_stage2['count_stage_2'].sum()
    "--------------"
    count_df['count_stage_1%'] = (count_df['count_stage_1'] / sum_of_pass_stage1_events) * 100
    count_df_stage2['count_stage_2%'] = (count_df_stage2['count_stage_2'] / sum_of_pass_stage2_events) * 100
    "--------------"

    "--------------"
    count_df['Per day'] = (count_df['count_stage_1'] / time_period_of_test)  # .apply(np.ceil)

    count_df_stage2['Per day'] = (count_df_stage2['count_stage_2'] / time_period_of_test)  # .apply(np.ceil)
    return count_df, count_df_stage2


# Press the green button in the gutter to run the script.
def handle_click(trace, points, state):
    print(points.point_inds)


if __name__ == '__main__':
    os.system("taskkill /im chrome.exe /f")
    input_file_path = r'C:\Users\eddy.a\Documents\Python Scripts\False Alarm Test Analysis\Files_2\_Input_2.csv'
    results_path = r"C:\Users\eddy.a\Documents\Python Scripts\False Alarm Test Analysis\Files_2"

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    txt = results_path + '\\'
    ##change those params before start
    date = '13.01.21-03.02.21'
    time_period_of_test = 28  # 4 weeks of data
    number_of_invertes_in_fleet = 8115
    number_of_UPDATED_AFCI = 8115
    Arc_power_diff_th = 1000
    MaxPhaseshift_th = 4
    Amp_change_diff_th = 800
    Amp_change_percentage_th = 10

    signal_df = pd.read_csv(input_file_path)
    signal_df["Time"] = signal_df['TELEM DATE'] + " " + signal_df['TELEM TIME']
    date_format = '%d/%m/%Y %I:%M:%S %p'
    signal_df["Time"] = pd.to_datetime(signal_df["Time"], format=date_format)
    signal_df.sort_values(by='Time', inplace=True)
    signal_df.reset_index(drop=True, inplace=True)

    "filter stage 1 events: 18xF"
    filterfor_18xD = signal_df['Error Code'] == '18xD'
    filterfor_18x100 = signal_df['Error Code'] == '18x100'
    filterfor_18xBA = signal_df['Error Code'] == '18xBA'
    # df_18xD = signal_df[(signal_df['Error Code'] == '18xD') | (signal_df['Error Code'] == '18x100')]
    df_18xD = signal_df[filterfor_18xD | filterfor_18x100 | filterfor_18xBA].reset_index(drop=True)
    "filter stage 2 events: 18xD"
    df_18xF = signal_df.loc[lambda x: x['Error Code'] == '18xF'].reset_index(drop=True)
    "Renaming df"
    df_18xF.rename(columns={'Param1Float': 'Arc_power_diff', 'Param2Float': 'Bitmap', 'Param3Float': 'Max_Phase_shift',
                            'Param4Float': 'Amp_change_diff', 'Param5Float': 'Amp_change_percentage',
                            'TELEM DATE': 'TELEM_DATE', 'TELEM TIME': 'TELEM_TIME', 'Error Code': 'Error_Code'},
                   inplace=True)

    df_18xD.rename(
        columns={'Param1Float': 'Mngr_pwr', 'Param2Float': 'ArcMaxEnergy', 'Param3Float': 'ArcMaxCurrentDrop',
                 'Param4Float': 'ArcAverageIac', 'Param5Float': 'ArcminEnergy',
                 'TELEM DATE': 'TELEM_DATE', 'TELEM TIME': 'TELEM_TIME', 'Error Code': 'Error_Code'},
        inplace=True)

    "Bulid dict for stage1 and stage2, key is the SN the value is class event"

    df_18xF_dict = dict()
    df_18xD_dict = dict()
    for row in df_18xF.iterrows():
        new_event = F_Event(row[1].TELEM_DATE, row[1].TELEM_TIME, row[1].Error_Code, row[1].Arc_power_diff,
                            row[1].Bitmap, row[1].Max_Phase_shift, row[1].Amp_change_diff, row[1].Amp_change_percentage,
                            row[1].Time)
        if row[1].SN not in df_18xF_dict:

            df_18xF_dict[row[1].SN] = [new_event]
        else:
            df_18xF_dict[row[1].SN].append(new_event)
    df_18xF_dict_sorted_by_len = sort_by_values_len(df_18xF_dict)

    for row in df_18xD.iterrows():
        new_event = D_Event(row[1].TELEM_DATE, row[1].TELEM_TIME, row[1].Error_Code, row[1].Mngr_pwr,
                            row[1].ArcMaxEnergy, row[1].ArcMaxCurrentDrop, row[1].ArcAverageIac, row[1].ArcminEnergy,
                            row[1].Time)
        if row[1].SN not in df_18xD_dict:

            df_18xD_dict[row[1].SN] = [new_event]
        else:
            df_18xD_dict[row[1].SN].append(new_event)
    df_18xD_dict_sorted_by_len = sort_by_values_len(df_18xD_dict)

    count_df, count_df_stage2 = bulid_count(df_18xF, df_18xD)
    total_pass_stage1_inverters = len(count_df)
    total_pass_stage2_inverters = len(count_df_stage2)
    sum_of_pass_stage1_events = count_df['count_stage_1'].sum()  # also in bulid_count
    sum_of_pass_stage2_events = count_df_stage2['count_stage_2'].sum()  # also in bulid_count
    "ploting"
    min_precent = 5  # we will plot only inverter that casue 3% events (or bigger)
    pass_stage2_list = []
    for sn in df_18xF_dict_sorted_by_len:
        Arc_power_diff = []
        Max_Phase_shift = []
        Amp_change_percentage = []
        Amp_change_diff = []
        bitmap_list = []
        time = []
        Arc_power_diff_th_list = []
        MaxPhaseshift_th_list = []
        Amp_change_diff_th_list = []
        Amp_change_percentage_th_list = []
        temp = []
        for key, values in sn.items():  ##for each serial....
            pass_stage2_counter = 0
            if count_df['count_stage_1%'].loc[key] < min_precent:  # plot only inverters that pass 3% from all events
                continue
            for event in range(len(values)):  ##for each event that happend in this serial
                Arc_power_diff.append(values[event].Arc_power_diff)
                Max_Phase_shift.append(values[event].Max_Phase_shift)
                Amp_change_percentage.append(values[event].Amp_change_percentage)
                Amp_change_diff.append(values[event].Amp_change_diff)
                time.append(values[event].Date_hour)
                bitmap_list.append(values[event].Bitmap)
                pass_stage2_counter += values[event].pass_stage_2
                Arc_power_diff_th_list.append(Arc_power_diff_th)
                MaxPhaseshift_th_list.append(MaxPhaseshift_th)
                Amp_change_diff_th_list.append(Amp_change_diff_th)
                Amp_change_percentage_th_list.append(Amp_change_percentage_th)
                temp.append(values[event].Date_hour.date())
            pass_stage2_list.append(bitmap_list)

            "plots"
            # hist_per_day_real = Counter(time)
            # DF_hist = pd.DataFrame.from_dict(hist_per_day_real, orient='index')
            # fig = px.histogram(DF_hist, y=0, x=DF_hist.index, nbins=time_period_of_test)
            # plotly.offline.plot(fig,filename=results_path+'\\'+str(key)+' Events Real events per day'+date+'.html')
            plot_4_pane('Arc_power_diff', 'Max_Phase_shift', 'Amp_change_percentage', 'Amp_change_diff',
                        count_df['count_stage_1%'], key, Arc_power_diff, Max_Phase_shift, Amp_change_percentage,
                        Amp_change_diff, Arc_power_diff_th_list, MaxPhaseshift_th_list,
                        Amp_change_percentage_th_list, Amp_change_diff_th_list, time, bitmap_list, results_path)
            plot_4_histogram('Arc_power_diff', 'Max_Phase_shift', 'Amp_change_percentage', 'Amp_change_diff',
                             count_df['count_stage_1%'], key, Arc_power_diff, Max_Phase_shift, Amp_change_percentage,
                             Amp_change_diff, Arc_power_diff_th_list, MaxPhaseshift_th_list,
                             Amp_change_percentage_th_list, Amp_change_diff_th_list, time, bitmap_list, results_path)

    temp = []
    for sn in df_18xF_dict_sorted_by_len:
        print(sn)
        for key, values in sn.items():
            pass_stage2_counter = 0
            for event in range(len(values)):  ##for each event that happend in this serial
                temp.append((key, values[event].Date_hour.date()))
    DF_hist = pd.Series(temp)
    Stage_1_Real_Events = DF_hist.value_counts()
    fig = go.Figure()
    fig = px.histogram(Stage_1_Real_Events, nbins=1000)
    fig.update_layout(
        title='Stage 1->Real Events  events per day' + "<br>" + 'Maximum number of events per day is ' + str(
            max(DF_hist.value_counts())))
    fig.update_xaxes(title_text='Real events per day')
    fig.update_yaxes(title_text='inv count')

    plotly.offline.plot(fig, filename=results_path + '\\' + ' Stage 1 Real Events  events per day' + date + '.html')

    for sn in df_18xD_dict_sorted_by_len:
        Mngr_pwr = []
        ArcMaxEnergy = []
        ArcMaxCurrentDrop = []
        ArcAverageIac = []
        ArcminEnergy = []
        time = []
        for key, values in sn.items():
            if count_df_stage2['count_stage_2%'].loc[key] < min_precent:
                continue
            for event in range(len(values)):
                Mngr_pwr.append(values[event].Mngr_pwr)
                ArcMaxEnergy.append(values[event].ArcMaxEnergy)
                ArcMaxCurrentDrop.append(values[event].ArcMaxCurrentDrop)
                ArcAverageIac.append(values[event].ArcAverageIac)
                ArcminEnergy.append(values[event].ArcminEnergy)
                time.append(values[event].Date_hour)
            "plot"
            #
            # plot_4_histogram_D('ArcMaxEnergy', 'ArcMaxCurrentDrop', 'ArcAverageIac', 'ArcminEnergy',
            #         count_df_stage2['count_stage_2%'] ,key,ArcMaxEnergy,ArcMaxCurrentDrop,ArcAverageIac, ArcminEnergy, time,len(values),results_path)

    del DF_hist
    temp = []
    for sn in df_18xD_dict_sorted_by_len:
        for key, values in sn.items():  ##for each serial....
            pass_stage2_counter = 0
            for event in range(len(values)):  ##for each event that happend in this serial
                temp.append((key, values[event].Date_hour.date()))
    DF_hist = pd.Series(temp)
    Stage_2_Real_Events = DF_hist.value_counts()
    fig = go.Figure()
    fig = px.histogram(Stage_2_Real_Events, nbins=50)
    fig.update_layout(
        title='Stage 2->Real Events  events per day' + "<br>" + 'Maximum number of events per day is ' + str(
            max(DF_hist.value_counts())))
    fig.update_xaxes(title_text='Real events per day')
    fig.update_yaxes(title_text='inv count')
    fig.update_traces(marker_color='red')
    plotly.offline.plot(fig, filename=results_path + '\\' + 'Stage 2 Real Events  events per day' + date + '.html')

    plot_avg_events_per_day(count_df, count_df_stage2, results_path, date, Stage_1_Real_Events, Stage_2_Real_Events)
    summary_pie_plot(count_df, count_df_stage2, results_path, date)
    All_data_stage_1_histograms(df_18xF, Arc_power_diff_th, MaxPhaseshift_th, Amp_change_percentage_th,
                                Amp_change_diff_th, bitmap_list, results_path, date)
    All_data_stage_2_histograms(df_18xD, results_path, date)
    path = os.path.realpath(results_path)
    os.startfile(path)
    print_hi('open result file')
