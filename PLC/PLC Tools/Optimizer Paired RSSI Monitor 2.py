import math
import os
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
date_format = "%d/%m/%y %H:%M"
date_time_index = [9, 11]   # take only the hours
date_date_index = [0, 8]    # take the whole date - not working
replace_nans_and_infs = [True, -1, 666666]
sorting_order_main = [True, ['Optimizer ID', 'Portia ID', 'Date']]      # for the main DF only!
rename_df_columns = {"siteid": "Site ID", "deviceid": "Portia ID", "managerid": "Manager ID", "get_time": "Date", "optimizerid": "Optimizer ID",
                     "param_129": "Last RSSI", "param_740": "Last SNR", "param_141": "Paired RSSI", "param_157": "Upper Ratio limit", "param_142": "Lower Ratio limit"}


# ## Plotting:
rssi_true__or__ratio_false = False
sorting_order_plots = [True, ['Optimizer ID', 'Portia ID'], 'Date']      # This will make the plotting VERY SLOW!
remove_glitches = [False, 1.5, 0.6667]
split_figs = 60
plot_addresses = {}


def round_numbers(df):
    how_to_round = {"Last RSSI": 0, "Paired RSSI": 0, "RSSI Ratio": 2, "Upper Ratio limit": 2, "Lower Ratio limit": 2, "Upper RSSI limit": 0, "Lower RSSI limit": 0, "Last SNR": 2}
    df = df.round(how_to_round)
    for data_set, round_number in how_to_round.items():
        if round_number == 0:
            df[data_set] = df[data_set].apply(int)
    return df


def remove_glitches_algorithm(sdf):
    average = sdf["RSSI Ratio"].mean()
    upper = average * remove_glitches[1]
    lower = average * remove_glitches[2]
    return [r if lower < r < upper else None for r in list(sdf["RSSI Ratio"])]


def concat_dfs(file_in_1, file_in_2, file_out):
    df1 = pd.read_csv(file_in_1).dropna(how='all', axis='columns')
    df2 = pd.read_csv(file_in_2).dropna(how='all', axis='columns')
    df = pd.concat([df1, df2])
    df.drop_duplicates(keep='last', inplace=True)
    df.to_csv(file_out, index=False)


def combine_df_and_p141(file_in_1, file_in_2, file_out):
    print(f'\nReading df1 ({file_in_1})')
    df1 = pd.read_csv(file_in_1).dropna(how='all', axis='columns')       # converters={'optimizerid_hex': partial(int, base=16)}
    df1 = df1.rename(columns=rename_df_columns)
    df1["Optimizer ID"] = df1["Optimizer ID"].apply(lambda n: f'{n:X}')      # df['optimizerid'].apply(hex)
    df1["Date"] = pd.to_datetime(df1["Date"]).dt.strftime(date_format)
    df1.drop(columns=[col for col in df1 if col not in rename_df_columns.values()], inplace=True)
    print(f'{list(df1.columns) = }')
    print(f'{df1.shape = }')

    print(f'\nReading df2 ({file_in_2})')
    df2 = pd.read_csv(file_in_2).dropna(how='all', axis='columns')       # converters={'optimizerid_hex': partial(int, base=16)}
    df2 = df2.rename(columns=rename_df_columns)
    df2["Optimizer ID"] = df2["Optimizer ID"].apply(lambda n: f'{n:X}')
    print(f'{list(df2.columns) = }')
    print(f'{df2.shape = }')

    print(f'\nMerging df1 and df2 ({file_out})')
    df3 = pd.merge(df1, df2, on=["Optimizer ID", "Portia ID"])
    print(f'{list(df3.columns) = }')
    print(f'{df3.shape = }\n')

    if df3.isnull().values.any():
        print('ERROR! There are some Nones in the data frame')
        print(f'{df3.isnull() = }')
        return None
    else:
        df3["Portia ID"] = df3["Portia ID"].apply(lambda n: f'{n:X}')
        df3["Manager ID"] = df3["Manager ID"].apply(lambda n: f'{n:X}')
        df3["RSSI Ratio"] = df3["Last RSSI"] / df3["Paired RSSI"]
        df3["Upper RSSI limit"] = df3["Paired RSSI"] * df3["Upper Ratio limit"]
        df3["Lower RSSI limit"] = df3["Paired RSSI"] / df3["Lower Ratio limit"]
        df_full = df3

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
    print(f'plot_df(): {file_out = }, {split_figs = }, {auto_open_html = }')
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
        sdf = df[df["Optimizer ID"] == optimizer_id]
        plot_title = f'Optimizer {optimizer_id}, L/U Ratios = {sdf["Lower Ratio limit"].iloc[0]} / {sdf["Upper Ratio limit"].iloc[0]} (Inverter {sdf["Portia ID"].iloc[0]} / Site {sdf["Site ID"].iloc[0]})'
        if optimizer_index % split_figs == 0:
            fig_title = plot_title

        if rssi_true__or__ratio_false:
            plots = ["Last RSSI"]
            for trace in plots:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=optimizer_index % split_figs == 0,
                                         hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace + ': %{y}<extra></extra>'), col=1, row=1)
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
            plots = ["RSSI Ratio"]
            for trace in plots:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=visible, mode='lines+markers', line=line, text=hour,
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
            fig.write_html(f"{file_out[:-5]} {fig_count:02} - Optimizers {optimizer_index - (optimizer_index % split_figs) + 1} to {optimizer_index + 1}.{file_out[-4:]}", auto_open=auto_open_html)


def summarize(df):
    global plot_addresses
    df_full = pd.DataFrame()
    for optimizer_index, optimizer_id in enumerate(df["Optimizer ID"].unique()):
        print(f'Summarizing Optimizer number {optimizer_index + 1}: {optimizer_id = }')
        sdf = df[df["Optimizer ID"] == optimizer_id]
        ssdf = sdf.iloc[-1:].rename(columns={'RSSI Ratio': 'Last Ratio'})
        ssdf["RSSI Average"] = sdf["Last RSSI"].mean()
        ssdf["Ratio Average"] = sdf["RSSI Ratio"].mean()
        ssdf["Samples above Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Upper Ratio limit']) if n > l])
        ssdf["Samples below Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Lower Ratio limit']) if n < (1 / l)])
        if len(plot_addresses) > 0:
            ssdf["Optimizer Index"] = plot_addresses[optimizer_id]["Optimizer Index"]
            ssdf["Fig Index"] = plot_addresses[optimizer_id]["Fig Index"]
            ssdf["Plot Index"] = plot_addresses[optimizer_id]["Plot Index"]
        df_full = pd.concat([df_full, ssdf], axis=0)
    if round_df_numbers:
        df_full = round_numbers(df_full)
    return df_full


def plot_histogram(main_df, summary_df, file_out):
    print(f'plot_histogram(): {file_out = }, {auto_open_html = }')

    # ## fig 01:
    fig_title = "Optimizers within, above, or below the Ratios"
    fig = make_subplots(rows=1, cols=1)
    bins = [0, 1, 10, 100, 1000, 10000]
    bins_labels = ['No samples above or below the ratios', '1-10', '10-100', '100-1000', '1000+']
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    for index, value_type in enumerate(value_types):
        for col in ['Samples above Ratio', 'Samples below Ratio']:
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

    # fig 02:
    # fig_title = "RSSI change per Day"
    # main_df.loc[:, "Date"] = main_df.index.map(lambda s: s[date_date_index[0]:date_date_index[1]])
    # fig = make_subplots(rows=1, cols=1)
    # bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100000]
    # bins_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 2)] + [f'{bins[-2]}+']
    # value_types = ["Absolute", "Percentage"]
    # vals = {value_types[0]: [], value_types[1]: []}
    rssi_change_per_year = list()
    rssi_change_per_day = list()
    # for optimizer_index, optimizer_id in enumerate(main_df["Optimizer ID"].unique()):
    #     try:
    #         temp_df = main_df[main_df["Optimizer ID"] == optimizer_id]
    #         date = temp_df["Date"].unique()[10]
    #         rssi_change_per_year.append(10 * math.log10(max(temp_df["Last RSSI"]) / min(temp_df["Last RSSI"])))
    #         temp_df = temp_df[temp_df["Date"] == date]
    #         rssi_change_per_day.append(10 * math.log10(max(temp_df["Last RSSI"]) / min(temp_df["Last RSSI"])))
    #         print(f'Fig 02 Optimizer number {optimizer_index + 1}, {optimizer_id = }, {date = }')
    #     except:
    #         print(f'ERROR! Fig 02 Optimizer number {optimizer_index + 1}, {optimizer_id = }, {date = }: min() = 0')
    # for index, value_type in enumerate(value_types):
    #     hist, bins = np.histogram(rssi_change_per_year, bins=bins)
    #     if index == 1:
    #         hist = [h / sum(hist) * 100 for h in hist]
    #         hover_template = f'RSSI Change per Year<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
    #     else:
    #         hover_template = f'RSSI Change per Year<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
    #     vals[value_type].append(hist)
    #     fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Change per Year', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    #     hist, bins = np.histogram(rssi_change_per_day, bins=bins)
    #     if index == 1:
    #         hist = [h / sum(hist) * 100 for h in hist]
    #         hover_template = f'RSSI Change per Day<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
    #     else:
    #         hover_template = f'RSSI Change per Day<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
    #     vals[value_type].append(hist)
    #     fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Change per Day', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    # fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    # fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Ratio", yaxis_title="Events or Percentage")
    # fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    # fig.update_layout(updatemenus=[
    #     dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    # plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 02 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)

    # Fig 03:
    fig_title = "RSSI change per Hour"
    main_df.loc[:, "Time"] = main_df.index.map(lambda s: s[date_time_index[0]:date_time_index[1]])
    fig = make_subplots(rows=1, cols=1)
    bins = sorted([int(t) for t in main_df["Time"].unique()])
    ratios = [[0, 0.4], [0.4, 0.75], [0.75, 1.5], [1.5, 5], [5, 10000]]
    bins_labels = [f'{b}:00-{b}:59' for b in bins]
    ratios_labels = [f'Ratio between {l} and {h}' for l, h in ratios]
    value_types = ["Absolute", "Percentage"]
    vals = {value_types[0]: [], value_types[1]: []}
    for index, value_type in enumerate(value_types):
        for ratio_i, (ratio_l, ratio_h) in enumerate(ratios):
            temp_df = main_df[(main_df["RSSI Ratio"] >= ratio_l) & (main_df["RSSI Ratio"] < ratio_h)]
            hist, bins = np.histogram(temp_df["Time"].astype(int).apply(lambda n: abs(n)), bins=bins)
            if index == 1:
                hist = [h / sum(hist) * 100 for h in hist]
                hover_template = f'RSSI Ratio at {ratios_labels[ratio_i]}<br>' + 'x: %{x}<br>' + 'y: %{y:.2f}<extra></extra>'
            else:
                hover_template = f'RSSI Ratio at {ratios_labels[ratio_i]}<br>' + 'x: %{x}<br>' + 'y: %{y}<extra></extra>'
            vals[value_type].append(hist)
            fig.add_trace(go.Bar(y=hist, x=bins_labels, name=f'RSSI Ratio at {ratios_labels[ratio_i]}', visible=index == 0, hovertemplate=hover_template), row=1, col=1)
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(title=fig_title, title_font_color="#407294", title_font_size=40, legend_title="Plots:", xaxis_title="Time of day", yaxis_title="Events or Percentage")
    fig.update_traces(marker_line_color='black', marker_line_width=0.5)
    fig.update_layout(updatemenus=[
        dict(buttons=[dict(label=f'Show {vt}', method='restyle', args=['y', [d for d in vals[vt]]]) for vt in value_types], direction="right", pad={"r": 10, "t": 50}, showactive=True, x=0.11, xanchor="right", y=1.12, yanchor="top"), ])
    plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{file_out[:-5]} 03 - {fig_title}{file_out[-5:]}', auto_open=auto_open_html)


if __name__ == "__main__":
    folder = r"M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy\RSSI Ratio - Meyer Burger modules\CSVs"
    file_path_in = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06.csv"
    file_path_full = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06 - RAW data.csv"
    file_path_output_1 = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06 - Optimizer RSSI Monitor.html"
    file_path_output_2 = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06 - Optimizer RSSI Histogram.html"
    file_path_output_summary = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06 - Summary.csv"

    combine_two_RAW_files_together = False
    file_path_in_1 = f"{folder}\\Ratio_Meyer_Burger_modules_2024_10_06.csv"
    file_path_in_2 = f"{folder}\\Ratio_Meyer_Burger_modules_2.csv"

    T_read__or__F_write = True
    file_path_p141 = f"{folder}\\Optimizer Paired RSSI and Limits.csv"

    print_all_optimizers = False
    print_all_histograms = True

    # Process:
    if combine_two_RAW_files_together:
        concat_dfs(file_path_in_1, file_path_in_2, file_path_in)        # Concatenate two CSVs of monitoring files

    if T_read__or__F_write:
        main_df = pd.read_csv(file_path_full, index_col=0).dropna(how='all', axis='columns')
        summary_df = pd.read_csv(file_path_output_summary).dropna(how='all', axis='columns')
    else:
        main_df = combine_df_and_p141(file_path_in, file_path_p141, file_path_full)     # Combine the monitoring file with the "static" parameters (like P141)
        summary_df = summarize(main_df)
        summary_df.to_csv(file_path_output_summary, index=False)

    if print_all_optimizers:
        plot_df(main_df, file_path_output_1)

    if print_all_histograms:
        plot_histogram(main_df, summary_df, file_path_output_2)
