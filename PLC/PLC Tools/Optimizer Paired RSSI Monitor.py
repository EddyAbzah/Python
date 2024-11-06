import os
import sys
import math
import statistics
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


# ## ### True ### ## # ## ### False ### ## #
# ## DataFrame:
auto_open_html = False
round_df_numbers = True
replace_nans_and_infs = [True, -1, 666666]
sorting_order_main = [True, ['Optimizer ID', 'Portia ID', 'Date']]  # for the main DF only!
rename_df_columns = {"deviceid": "Portia ID", "managerid": "Manager ID", "time": "Date", "OPT ID": "Optimizer ID",
                     "param_129": "Last RSSI", "param_141": "Paired RSSI", "param_157": "Upper RSSI limit", "param_142": "Lower RSSI limit"}

# ## Algorithm:
rssi_ratio_algorithm_enable = [True, 2]
rssi_original_ratio = [2.51188643150958, 0.3981071705534972, 1]

# ## Plotting:
rssi_true__or__ratio_false = False
sorting_order_plots = [True, ['Optimizer ID', 'Portia ID'], 'Date']  # This will make the plotting VERY SLOW!
plot_old_limits = False
remove_glitches = [False, 1.5, 0.6667]
split_figs = 100
plot_addresses = {}

# ## Terminal output:
output_text = True
output_text_path = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Analysis from 30-08-2023 to 15-10-2024\Python log.txt"


def round_numbers(df):
    how_to_round = {"Last RSSI": 0, "Paired RSSI": 0, "RSSI Ratio": 2, "Upper Ratio limit": 2, "Lower Ratio limit": 2, "Old Upper Ratio limit": 2, "Last RSSI Diff": 0, "RSSI Ratio Diff": 2,
                    "Old Lower Ratio limit": 2, "New Upper Ratio limit": 2, "New Lower Ratio limit": 2, "New C Ratio": 2, "Ratio Average": 2, "RSSI Standard Deviation": 0, "Ratio Standard Deviation": 2, "New vs Old Ratio": 2}
    df = df.round(how_to_round)
    for data_set, round_number in how_to_round.items():
        if round_number == 0 and data_set in df:
            df[data_set] = df[data_set].apply(int)
    return df


def remove_glitches_algorithm(sdf):
    average = sdf["RSSI Ratio"].mean()
    upper = average * remove_glitches[1]
    lower = average * remove_glitches[2]
    return [r if lower < r < upper else None for r in list(sdf["RSSI Ratio"])]


def rssi_ratio_algorithm(sdf, data):
    delay = rssi_ratio_algorithm_enable[1]
    upper = [rssi_original_ratio[0]]
    lower = [rssi_original_ratio[1]]
    c_ratio = [rssi_original_ratio[2]]
    ka_timeouts = [0]
    data = list(sdf[data][:-1])
    index = 0
    while index < len(data):
        if data[index] == 0 or index + delay >= len(data) or any([lower[index] < data[i] < upper[index] for i in range(index, index + delay)]):
            c_ratio.append(c_ratio[index])
            upper.append(upper[index])
            lower.append(lower[index])
            ka_timeouts.append(ka_timeouts[index])
        else:
            c = (data[index] + rssi_original_ratio[2]) / 2
            c_ratio.append(c)
            upper.append(rssi_original_ratio[0] * c)  # bug = upper.append(upper[index] * c)
            lower.append(rssi_original_ratio[1] * c)  # bug = lower.append(lower[index] * c)
            ka_timeouts.append(ka_timeouts[index] + 1)
        index += 1
    return pd.concat([sdf, pd.DataFrame({"New Upper Ratio limit": upper, "New Lower Ratio limit": lower, "New C Ratio": c_ratio, "Ratio Adjustments": ka_timeouts}, index=sdf.index)], axis=1)


def concat_dfs(files_in, file_out, remove_duplicates):
    dfs = []
    for file in files_in:
        dfs.append(pd.read_csv(file).dropna(how='all', axis='columns'))
    df = pd.concat(dfs)
    if remove_duplicates:
        df.drop_duplicates(keep='last', inplace=True)
    df.to_csv(file_out, index=False)


def combine_df_and_p141(file_in_1, file_in_2, file_out):
    print(f'\nReading df1 ({file_in_1})')
    df1 = pd.read_csv(file_in_1).dropna(how='all', axis='columns')  # converters={'optimizerid_hex': partial(int, base=16)}
    df1['Optimizer ID'] = df1['optimizerid'].apply(lambda n: f'{n:X}')  # df['optimizerid'].apply(hex)
    df1['time'] = pd.to_datetime(df1['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df1 = df1.rename(columns=rename_df_columns)
    df1.drop(columns=[col for col in df1 if col not in rename_df_columns.values()], inplace=True)
    print(f'{list(df1.columns) = }')
    print(f'{df1.shape = }')

    print(f'\nReading df2 ({file_in_2})')
    df2 = pd.read_csv(file_in_2).dropna(how='all', axis='columns')  # converters={'optimizerid_hex': partial(int, base=16)}
    df2 = df2.rename(columns=rename_df_columns)
    print(f'{list(df2.columns) = }')
    print(f'{df2.shape = }')

    print(f'\nMerging df1 and df2 ({file_out})')
    df3 = pd.merge(df1, df2, on="Optimizer ID")
    print(f'{list(df3.columns) = }')
    print(f'{df3.shape = }\n')

    if len(df3) < len(df1):
        print(f'ERROR! {len(df3) = } is smaller than {len(df1) = }... EXIT!!!')
        exit()
    elif any((df3['Manager ID'].sort_values() - df3['Portia ID'].sort_values()).diff()[1:] != 0):
        print('ERROR! There is a mismatch in Inverter IDs dec vs. hex')
        print(f'{df3['Manager ID'].nunique() = }')
        print(f'{df3['Portia ID'].nunique() = }')
        return None
    elif df3.isnull().values.any():
        print('ERROR! There are some Nones in the data frame')
        print(f'{df3.isnull() = }')
        return None
    else:
        df3["Portia ID"] = df3["Portia ID"].apply(lambda n: f'{n:X}')
        df3["Manager ID"] = df3["Manager ID"].apply(lambda n: f'{n:X}')
        df3["RSSI Ratio"] = df3["Last RSSI"] / df3["Paired RSSI"]
        df3["Old Upper Ratio limit"] = rssi_original_ratio[0]
        df3["Old Lower Ratio limit"] = rssi_original_ratio[1]
        df3["Old C Ratio"] = rssi_original_ratio[2]
        if not rssi_ratio_algorithm_enable[0]:
            df3["New Upper Ratio limit"] = rssi_original_ratio[0]
            df3["New Lower Ratio limit"] = rssi_original_ratio[1]
            df3["New C Ratio"] = rssi_original_ratio[2]
            df_full = df3
        else:
            df_full = pd.DataFrame()
            for optimizer_index, optimizer_id in enumerate(df3["Optimizer ID"].unique()):
                print(f'rssi_ratio_algorithm() for Optimizer number {optimizer_index + 1}: {optimizer_id = }: change ratio after {rssi_ratio_algorithm_enable[1]} KA timeouts)')
                temp_df = df3[df3["Optimizer ID"] == optimizer_id].sort_values("Date")
                if remove_glitches[0]:
                    temp_df.loc[:, "RSSI Ratio"] = remove_glitches_algorithm(temp_df)
                    temp_df = temp_df.dropna(axis=0)
                temp_df = rssi_ratio_algorithm(temp_df, "RSSI Ratio")
                temp_df["Last RSSI Diff"] = temp_df["Last RSSI"] - temp_df["Last RSSI"].mean()
                temp_df["RSSI Ratio Diff"] = temp_df["RSSI Ratio"] - temp_df["RSSI Ratio"].mean()
                df_full = pd.concat([df_full, temp_df], axis=0)

        del df1
        del df2
        del df3
        if round_df_numbers:
            df_full = round_numbers(df_full)
        for col in df_full.columns:
            print(f'df_full[{col}].nunique() = {df_full[col].nunique()}')
        if sorting_order_main[0]:
            print(f'\nSorting values by {sorting_order_main[1]} before export')
            for sort_by in sorting_order_main[1]:
                df_full.sort_values(sort_by, inplace=True)

        print(f'\nFinal df = df_full')
        print(f'{list(df_full.columns) = }')
        print(f'{df_full.shape = }')
        df_full.set_index('Date', inplace=True)
        df_full.to_csv(file_out)
        return df_full


def plot_df(df, file_out):
    global plot_addresses
    print(f'plot_df(): {file_out = }, {split_figs = }')
    fig_count = 0
    print(f'{df["Optimizer ID"].nunique() = }')
    if sorting_order_plots[0]:
        print(f'\nSorting values by {sorting_order_plots[1]} and then by {sorting_order_plots[2]} before plot')
        for sort_by in sorting_order_plots[1]:
            df.sort_values(sort_by, inplace=True)
    for optimizer_index, optimizer_id in enumerate(df["Optimizer ID"].unique()):
        if optimizer_index % split_figs == 0:
            fig = make_subplots(cols=1, rows=1, shared_xaxes=False)
            steps = list()
            fig_count += 1
        print(f'Plotting Optimizer number {optimizer_index + 1} in fig number {fig_count}, place {int(optimizer_index % split_figs) + 1}: {optimizer_id = }')
        if sorting_order_plots[0]:
            df.sort_values(sorting_order_plots[2], inplace=True)
        plot_addresses.update({optimizer_id: {"Optimizer Index": optimizer_index + 1, "Fig Index": fig_count, "Plot Index": int(optimizer_index % split_figs) + 1}})
        sdf = df[df["Optimizer ID"] == optimizer_id].sort_values("Date")
        plot_title = f'Optimizer {optimizer_id} (Inverter {sdf["Portia ID"].iloc[0]} / {sdf["Manager ID"].iloc[0]}): Ratio Adjustments = {sdf["Ratio Adjustments"].iloc[-1]}'
        if optimizer_index % split_figs == 0:
            fig_title = plot_title

        if rssi_true__or__ratio_false:
            if plot_old_limits:
                plots = ["Last RSSI", "Paired RSSI"]
            else:
                plots = ["Last RSSI"]
            for trace in plots:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=optimizer_index % split_figs == 0,
                                         hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace + ': %{y}<extra></extra>'), col=1, row=1)
            if plot_old_limits:
                for trace1, trace2, name in [("Old Upper Ratio limit", "Paired RSSI", "Old Upper Paired RSSI"), ("Old Lower Ratio limit", "Paired RSSI", "Old Lower Paired RSSI")]:
                    fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace1] * sdf[trace2], name=f"{optimizer_id} - {name}", visible=optimizer_index % split_figs == 0,
                                             hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace1 + ': %{y}<extra></extra>'), col=1, row=1)
            if rssi_ratio_algorithm_enable[0]:
                for trace1, trace2, name in [("New Upper Ratio limit", "Paired RSSI", "New Upper Paired RSSI"), ("New Lower Ratio limit", "Paired RSSI", "New Lower Paired RSSI"), ("New C Ratio", "Paired RSSI", "New Paired RSSI")]:
                    fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace1] * sdf[trace2], name=f"{optimizer_id} - {trace1}", visible=optimizer_index % split_figs == 0,
                                             hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace1 + ': %{y}<extra></extra>'), col=1, row=1)
        else:
            visible = optimizer_index % split_figs == 0
            hover_label = {"font_size": 14, "namelength": -1}
            time_is_not_visible = False
            if time_is_not_visible:
                hover_template = [f'<b>Optimizer HEX ID: {optimizer_id}</b><extra></extra><br><br>', ': %{y}<br>Date: %{x:%Y-%m-%d %H:%M:%S}<br>Time: %{text}']
                hour = [h.split(' ')[-1] for h in sdf.index]
            else:
                hover_template = [f'<b>Optimizer HEX ID: {optimizer_id}</b><extra></extra><br><br>', ': %{y}<br>Date: %{x:%Y-%m-%d %H:%M:%S}']
                hour = None
            marker_color = [int(h.split(' ')[-1][:2]) for h in sdf.index]
            line = dict(dash='dot', color='darkturquoise')
            if plot_old_limits:
                plots = ["RSSI Ratio", "Old C Ratio", "Old Upper Ratio limit", "Old Lower Ratio limit"]
            else:
                plots = ["RSSI Ratio"]
            for trace in plots:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=visible, mode='lines+markers', line=line, text=hour,
                                         hovertemplate=hover_template[0] + f'{trace}' + hover_template[1], hoverlabel=hover_label, marker_color=marker_color), col=1, row=1)
            if rssi_ratio_algorithm_enable[0]:
                for trace in ["New Upper Ratio limit", "New Lower Ratio limit", "New C Ratio"]:
                    fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=visible, text=hour,
                                             hovertemplate=hover_template[0] + f'{trace}' + hover_template[1], hoverlabel=hover_label, marker_color=marker_color), col=1, row=1)

        if optimizer_index % split_figs == 0:
            figs_per_fig = len(fig.data)
        step = dict(label=f'{int(optimizer_index % split_figs) + 1:02} Opt: {optimizer_id}', method="update",
                    args=[{"visible": [False] * figs_per_fig * split_figs}, {"title": plot_title}])
        step["args"][0]["visible"][optimizer_index % split_figs * figs_per_fig:optimizer_index % split_figs * figs_per_fig + figs_per_fig] = [True] * figs_per_fig
        steps.append(step)

        if (optimizer_index + 1) % split_figs == 0 or optimizer_index == df["Optimizer ID"].nunique() - 1:
            fig.update_layout(title=fig_title, title_font_color="#2589BB", title_font_size=40, legend_title="Traces:", legend_title_font_color="#2589BB")
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=steps, currentvalue={"prefix": "Plot number "})])
            fig.write_html(f"{file_out[:-5]} {fig_count:02} - Optimizers {optimizer_index - (optimizer_index % split_figs) + 1} to {optimizer_index + 1}.{file_out[-4:]}", auto_open=auto_open_html, config={'scrollZoom': True, 'editable': True})


def summarize(df):
    global plot_addresses
    df_full = pd.DataFrame()
    df.drop(columns=['Old C Ratio'], inplace=True)
    for optimizer_index, optimizer_id in enumerate(df["Optimizer ID"].unique()):
        print(f'Summarizing Optimizer number {optimizer_index + 1}: {optimizer_id = }')
        sdf = df[df["Optimizer ID"] == optimizer_id].sort_values("Date")
        ssdf = sdf.iloc[-1:].rename(columns={'RSSI Ratio': 'Last Ratio'})
        ssdf["RSSI Average"] = sdf["Last RSSI"].mean()
        ssdf["Ratio Average"] = sdf["RSSI Ratio"].mean()
        ssdf["RSSI Standard Deviation"] = statistics.stdev(sdf["Last RSSI"])
        ssdf["Ratio Standard Deviation"] = statistics.stdev(sdf["RSSI Ratio"])
        ssdf["Samples above Old Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Old Upper Ratio limit']) if n > l])
        ssdf["Samples below Old Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Old Lower Ratio limit']) if n < l])
        ssdf["Samples above New Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['New Upper Ratio limit']) if n > l])
        ssdf["Samples below New Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['New Lower Ratio limit']) if n < l])
        ssdf["New vs Old Ratio"] = (ssdf["Samples above Old Ratio"] + ssdf["Samples below Old Ratio"]) / (ssdf["Samples above New Ratio"] + ssdf["Samples below New Ratio"])
        if len(plot_addresses) > 0:
            ssdf["Optimizer Index"] = plot_addresses[optimizer_id]["Optimizer Index"]
            ssdf["Fig Index"] = plot_addresses[optimizer_id]["Fig Index"]
            ssdf["Plot Index"] = plot_addresses[optimizer_id]["Plot Index"]
        df_full = pd.concat([df_full, ssdf], axis=0)
    if replace_nans_and_infs[0]:
        df_full['New vs Old Ratio'] = df_full['New vs Old Ratio'].fillna(replace_nans_and_infs[1])
        df_full['New vs Old Ratio'].replace([np.inf, -np.inf], replace_nans_and_infs[2], inplace=True)
    df_full = df_full.sort_values(by=['New vs Old Ratio'], ascending=False)
    if round_df_numbers:
        df_full = round_numbers(df_full)
    return df_full


def plot_histogram(main_df, summary_df, file_out):
    print(f'plot_histogram(): {file_out = }')

    # ## fig 01:
    fig_title = "Samples above or below Old and New Ratios"
    fig = make_subplots(rows=1, cols=1)
    bins = [0, 1, 10, 100, 1000, 10000]
    bins_labels = ['0', '1-10', '10-100', '100-1000', '1000+']
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    for index, value_type in enumerate(value_types):
        for col in ['Samples above Old Ratio', 'Samples below Old Ratio', 'Samples above New Ratio', 'Samples below New Ratio']:
            hist, bins = np.histogram(summary_df[col], bins=bins)
            if index == 1:
                hist = [h / sum(hist) * 100 for h in hist]
                hover_template = f'{col}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
            else:
                hover_template = f'{col}<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
            vals[value_type].append(hist)
            fig.add_trace(go.Bar(y=hist, x=bins_labels, name=col, visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Samples", yaxis_title="Events or Percentage")
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(updatemenus=[dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 01 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)

    # ## fig 02:
    fig_title = "New Ratios and amount of changes"
    fig = make_subplots(rows=1, cols=1)
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 100, 10000]
    bins_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-5', '5-10', '10-100', '100-1000', '1000+']
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    for index, value_type in enumerate(value_types):
        for col in ['Ratio Adjustments', 'New C Ratio', 'New Lower Ratio limit', 'New Upper Ratio limit']:
            hist, bins = np.histogram(summary_df[col], bins=bins)
            if index == 1:
                hist = [h / sum(hist) * 100 for h in hist]
                hover_template = f'{col}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
            else:
                hover_template = f'{col}<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
            vals[value_type].append(hist)
            fig.add_trace(go.Bar(y=hist, x=bins_labels, name=col, visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Ratio", yaxis_title="Events or Percentage")
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(updatemenus=[dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 02 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)

    # ## fig 03:
    fig_title = "RSSI change per Day"
    fig = make_subplots(rows=1, cols=1)
    bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100000]
    bins_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 2)] + [f'{bins[-2]}+']
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    main_df.loc[:, "Time"] = main_df.index.map(lambda s: s[11:13])
    main_df.loc[:, "Date"] = main_df.index.map(lambda s: s[:10])
    rssi_change_per_year = list()
    rssi_change_per_day = list()
    for optimizer_index, optimizer_id in enumerate(main_df["Optimizer ID"].unique()):
        temp_df = main_df[main_df["Optimizer ID"] == optimizer_id].sort_values("Date")
        rssi_change_per_year.append(10 * math.log10(temp_df["Last RSSI"].max() / temp_df["Last RSSI"][temp_df["Last RSSI"] != 0].min()))
        date = temp_df["Date"].unique()[1]  # pick the second day or use np.random.choice(temp_df["Date"].unique())
        temp_df = temp_df[temp_df["Date"] == date]
        rssi_change_per_day.append(10 * math.log10(temp_df["Last RSSI"].max() / temp_df["Last RSSI"][temp_df["Last RSSI"] != 0].min()))
        print(f'Fig 03 Optimizer number {optimizer_index + 1}, {optimizer_id = }, {date = }')
    for index, value_type in enumerate(value_types):
        hist, bins = np.histogram(rssi_change_per_year, bins=bins)
        if index == 1:
            hist = [h / sum(hist) * 100 for h in hist]
            hover_template = f'RSSI Change per Year<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
        else:
            hover_template = f'RSSI Change per Year<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
        vals[value_type].append(hist)
        fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Change per Year', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
        hist, bins = np.histogram(rssi_change_per_day, bins=bins)
        if index == 1:
            hist = [h / sum(hist) * 100 for h in hist]
            hover_template = f'RSSI Change per Day<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
        else:
            hover_template = f'RSSI Change per Day<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
        vals[value_type].append(hist)
        fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Change per Day', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Ratio", yaxis_title="Events or Percentage")
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(updatemenus=[dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 03 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)

    # ## fig 04:
    fig_title = "RSSI change per Hour"
    fig = make_subplots(rows=1, cols=1)
    bins = [0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 4, 5, 10, 100000]
    bins_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 2)] + [f'{bins[-2]}+']
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    for index, value_type in enumerate(value_types):
        for time in main_df["Time"].unique():
            temp_df = main_df[main_df["Time"] == time]
            hist, bins = np.histogram(temp_df["RSSI Ratio Diff"].apply(lambda n: abs(n)), bins=bins)
            if index == 1:
                hist = [h / sum(hist) * 100 for h in hist]
                hover_template = f'RSSI Ratio Diff at {time}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
            else:
                hover_template = f'RSSI Ratio Diff at {time}<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
            vals[value_type].append(hist)
            fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Ratio Diff at {time}', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Ratio", yaxis_title="Events or Percentage")
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(updatemenus=[
        dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 04 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)


if __name__ == "__main__":
    if output_text:
        default_stdout = sys.stdout
        sys.stdout = open(output_text_path, 'w')

    T_read__or__F_write = True
    combine_two_RAW_files_together = False
    folder = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Analysis from 30-08-2023 to 15-10-2024"
    # folder = r"C:\Users\eddy.a\Downloads\Solution Procedure"
    # folder = r"C:\Users\eddy.a\Downloads\Analysis from 30-08-2023 to 15-10-2024"

    sub_folder = "Optimizer plots\\"
    file_path_output_csv = f"{folder}\\Analysis with {rssi_ratio_algorithm_enable[1]} Sample Delay.csv"

    # ## pre-RAW analysis:
    if combine_two_RAW_files_together:
        concat_dfs([r"C:\Users\eddy.a\Downloads\Solution Procedure\236 Day Analysis\236 days - RAW data.csv",
                    r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data\New data 14-10-2024\01.02.2024 - 30.04.2024.csv",
                    r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data\New data 14-10-2024\01.05.2024 - 31.07.2024.csv",
                    r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data\New data 14-10-2024\01.08.2024 - 15.10.2024.csv"],
                   file_out=f"{folder}\\RAW data.csv", remove_duplicates=True)

    # ## RAW analysis:
    if T_read__or__F_write:
        main_df = pd.read_csv(file_path_output_csv, index_col=0).dropna(how='all', axis='columns')
    else:
        main_df = combine_df_and_p141(f"{folder}\\RAW data.csv", f"{folder}\\P141.csv", file_path_output_csv)
    if not os.path.exists(f"{folder}\\{sub_folder}"):
        os.makedirs(f"{folder}\\{sub_folder}")
    plot_df(main_df, file_out=f"{folder}\\{sub_folder}Optimizer RSSI Monitor.html")

    # ## Summary:
    if T_read__or__F_write:
        summary_df = pd.read_csv(f"{folder}\\Summary.csv").dropna(how='all', axis='columns')
    else:
        summary_df = summarize(main_df)
        summary_df.to_csv(f"{folder}\\Summary.csv", index=False)
    plot_histogram(main_df, summary_df, file_out=f"{folder}\\Optimizer RSSI Histogram.html")

    if output_text:
        sys.stdout.close()
        sys.stdout = default_stdout
