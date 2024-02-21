import os
import heartrate
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


# ## ### True ### ## # ## ### False ### ## #
# heartrate.trace(browser=True)

# ## DataFrame:
round_df_numbers = True
replace_nans_and_infs = [True, -1, 666666]
sorting_order = ['Portia ID', 'Optimizer ID', 'Date']
rename_df_columns = {"deviceid": "Portia ID", "managerid": "Manager ID", "time": "Date", "OPT ID": "Optimizer ID",
                     "param_129": "Last RSSI", "param_141": "Paired RSSI", "param_157": "Upper RSSI limit", "param_142": "Lower RSSI limit"}

# ## Algorithm:
rssi_ratio_algorithm_enable = [True, 2]
rssi_original_ratio = [2.51188643150958, 0.3981071705534972, 1]

# ## Plotting:
rssi_true__or__ratio_false = False
plot_old_limits = False
remove_glitches = [False, 1.5, 0.6667]
split_figs = 100


def round_numbers(df):
    how_to_round = {"Last RSSI": 0, "Paired RSSI": 0, "RSSI Ratio": 2, "Upper Ratio limit": 2, "Lower Ratio limit": 2, "Old Upper Ratio limit": 2,
                    "Old Lower Ratio limit": 2, "New Upper Ratio limit": 2, "New Lower Ratio limit": 2, "New C Ratio": 2, "Ratio Average": 2, "New vs Old Ratio": 2}
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


def rssi_ratio_algorithm(sdf, data):
    delay = rssi_ratio_algorithm_enable[1]
    upper = [rssi_original_ratio[0]]
    lower = [rssi_original_ratio[1]]
    c_ratio = [rssi_original_ratio[2]]
    ka_timeouts = [0]
    data = list(sdf[data][:-1])
    index = 0
    while index < len(data):
        if lower[index] < data[index] < upper[index] or any([lower[index] < data[i] < upper[index] for i in range(index, min(index + delay, len(data)))]):
            c_ratio.append(c_ratio[index])
            upper.append(upper[index])
            lower.append(lower[index])
            ka_timeouts.append(ka_timeouts[index])
        else:
            c = (data[index] + rssi_original_ratio[2]) / 2
            c_ratio.append(c)
            upper.append(rssi_original_ratio[0] * c)    # bug = upper.append(upper[index] * c)
            lower.append(rssi_original_ratio[1] * c)    # bug = lower.append(lower[index] * c)
            ka_timeouts.append(ka_timeouts[index] + 1)
        index += 1
    return pd.concat([sdf, pd.DataFrame({"New Upper Ratio limit": upper, "New Lower Ratio limit": lower, "New C Ratio": c_ratio, "KA Timeouts": ka_timeouts}, index=sdf.index)], axis=1)


def combine_dfs(file_in_1, file_in_2, file_out):
    print(f'\nReading df1 ({file_in_1})')
    df1 = pd.read_csv(file_in_1).dropna(how='all', axis='columns')       # converters={'optimizerid_hex': partial(int, base=16)}
    df1['Optimizer ID'] = df1['optimizerid'].apply(lambda n: f'{n:X}')      # df['optimizerid'].apply(hex)
    df1['time'] = pd.to_datetime(df1['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df1 = df1.rename(columns=rename_df_columns)
    df1.drop(columns=[col for col in df1 if col not in rename_df_columns.values()], inplace=True)
    print(f'{list(df1.columns) = }')
    print(f'{df1.shape = }')

    print(f'\nReading df2 ({file_in_2})')
    df2 = pd.read_csv(file_in_2).dropna(how='all', axis='columns')       # converters={'optimizerid_hex': partial(int, base=16)}
    df2 = df2.rename(columns=rename_df_columns)
    print(f'{list(df2.columns) = }')
    print(f'{df2.shape = }')

    print(f'\nMerging df1 and df2 ({file_out})')
    df3 = pd.merge(df1, df2, on="Optimizer ID")
    print(f'{list(df3.columns) = }')
    print(f'{df3.shape = }\n')

    if any((df3['Manager ID'].sort_values() - df3['Portia ID'].sort_values()).diff()[1:] != 0):
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
                temp_df = df3[df3["Optimizer ID"] == optimizer_id]
                if remove_glitches[0]:
                    temp_df.loc[:, "RSSI Ratio"] = remove_glitches_algorithm(temp_df)
                    temp_df = temp_df.dropna(axis=0)
                temp_df = rssi_ratio_algorithm(temp_df, "RSSI Ratio")
                df_full = pd.concat([df_full, temp_df], axis=0)

        del df1
        del df2
        del df3
        if round_df_numbers:
            df_full = round_numbers(df_full)
        for col in df_full.columns:
            print(f'df_full[{col}].nunique() = {df_full[col].nunique()}')
        if sorting_order is not None and len(sorting_order) > 0:
            print(f'\nSorting values by {sorting_order} before export')
            for sort_by in sorting_order:
                df_full.sort_values(sort_by, inplace=True)

        print(f'\nFinal df = df_full')
        print(f'{list(df_full.columns) = }')
        print(f'{df_full.shape = }')
        df_full.set_index('Date', inplace=True)
        df_full.to_csv(file_out)
        return df_full


def plot_df(df, file_out, auto_open_html=True):
    print(f'plot_df(): {file_out = }, {split_figs = }, {auto_open_html = }')
    fig_count = 0
    print(f'{df["Optimizer ID"].nunique() = }')
    for optimizer_index, optimizer_id in enumerate(df["Optimizer ID"].unique()):
        if optimizer_index % split_figs == 0:
            fig = make_subplots(cols=1, rows=1, shared_xaxes=False)
            steps = list()
            fig_count += 1
        print(f'Plotting Optimizer number {optimizer_index + 1} in fig number {fig_count}, place {int(optimizer_index % split_figs) + 1}: {optimizer_id = }')
        sdf = df[df["Optimizer ID"] == optimizer_id]
        plot_title = f'Optimizer {optimizer_id} (Inverter {sdf["Portia ID"].iloc[0]} / {sdf["Manager ID"].iloc[0]}): KA timeouts = {sdf["KA Timeouts"].iloc[-1]}'

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
            if plot_old_limits:
                plots = ["RSSI Ratio", "Old C Ratio", "Old Upper Ratio limit", "Old Lower Ratio limit"]
            else:
                plots = ["RSSI Ratio"]
            for trace in plots:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=optimizer_index % split_figs == 0,
                                         mode='lines+markers', marker_color=[int(h.split(' ')[-1][:2]) for h in sdf.index], line=dict(dash='dot', color='darkturquoise'),
                                         hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace + ': %{y}<extra></extra>'), col=1, row=1)
            if rssi_ratio_algorithm_enable[0]:
                for trace in ["New Upper Ratio limit", "New Lower Ratio limit", "New C Ratio"]:
                    fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=optimizer_index % split_figs == 0,
                                             hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace + ': %{y}<extra></extra>'), col=1, row=1)

        if optimizer_index % split_figs == 0:
            figs_per_fig = len(fig.data)
        step = dict(label=f'{int(optimizer_index % split_figs) + 1:02} Opt: {optimizer_id}', method="update",
                    args=[{"visible": [False] * figs_per_fig * split_figs}, {"title": plot_title}])
        step["args"][0]["visible"][optimizer_index % split_figs * figs_per_fig:optimizer_index % split_figs * figs_per_fig + figs_per_fig] = [True] * figs_per_fig
        steps.append(step)

        if (optimizer_index + 1) % split_figs == 0 or optimizer_index == df["Optimizer ID"].nunique() - 1:
            fig.update_layout(title=plot_title, title_font_color="#2589BB", title_font_size=40, legend_title="Traces:", legend_title_font_color="#2589BB")
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=steps)])
            if rssi_true__or__ratio_false:
                temp_name = file_out[:-5] + " (RSSI)"
            else:
                temp_name = file_out[:-5] + " (Ratio)"
            fig.write_html(f"{temp_name} {fig_count:02} - Optimizers {optimizer_index - split_figs + 2} to {optimizer_index + 1}.{file_out[-4:]}", auto_open=auto_open_html)


def summarize(df):
    df_full = pd.DataFrame()
    df.drop(columns=['Old C Ratio'], inplace=True)
    for optimizer_index, optimizer_id in enumerate(df["Optimizer ID"].unique()):
        print(f'Summarizing Optimizer number {optimizer_index + 1}: {optimizer_id = }')
        sdf = df[df["Optimizer ID"] == optimizer_id]
        ssdf = sdf.iloc[-1:].rename(columns={'RSSI Ratio': 'Last Ratio'})
        ssdf["RSSI Average"] = sdf["Last RSSI"].mean()
        ssdf["Ratio Average"] = sdf["RSSI Ratio"].mean()
        ssdf["Samples above Old Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Old Upper Ratio limit']) if n > l])
        ssdf["Samples below Old Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['Old Lower Ratio limit']) if n < l])
        ssdf["Samples above New Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['New Upper Ratio limit']) if n > l])
        ssdf["Samples below New Ratio"] = sum([1 for n, l in zip(sdf['RSSI Ratio'], sdf['New Lower Ratio limit']) if n < l])
        ssdf["New vs Old Ratio"] = (ssdf["Samples above Old Ratio"] + ssdf["Samples below Old Ratio"]) / (ssdf["Samples above New Ratio"] + ssdf["Samples below New Ratio"])
        df_full = pd.concat([df_full, ssdf], axis=0)
    if replace_nans_and_infs[0]:
        df_full['New vs Old Ratio'] = df_full['New vs Old Ratio'].fillna(replace_nans_and_infs[1])
        df_full['New vs Old Ratio'].replace([np.inf, -np.inf], replace_nans_and_infs[2], inplace=True)
    df_full = df_full.sort_values(by=['New vs Old Ratio'], ascending=False)      # 'KA Timeouts'
    if round_df_numbers:
        df_full = round_numbers(df_full)
    return df_full


def plot_histogram(df, file_out, auto_open_html=True, SEX=True):
    n_bins = 10
    filter_limits = [0, 10]
    print(f'plot_histogram(): {file_out = }, {n_bins = }, {auto_open_html = }')
    if not SEX:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    else:
        fig = make_subplots(cols=2, rows=2, shared_xaxes=False)

    data_1 = [n for n in df['New Upper Ratio limit'] if filter_limits[0] <= n <= filter_limits[1]]
    data_2 = [n for n in df['New Lower Ratio limit'] if filter_limits[0] <= n <= filter_limits[1]]
    data_3 = [n for n in df['New C Ratio'] if filter_limits[0] <= n < filter_limits[1]]
    if not SEX:
        ax1.hist([data_1, data_2, data_3], n_bins, density=True, histtype='bar', label=['Upper', 'Lower', 'C'])
        ax1.legend(prop={'size': 10})
        ax1.set_title('New Ratios after algorithm')
    else:
        fig.add_trace(go.Histogram(x=[data_1, data_2, data_3], histnorm='percent', name='New Ratios after algorithm', nbinsx=n_bins), col=1, row=1)

    data = [n for n in df['Ratio Average'] if filter_limits[0] <= n <= filter_limits[1]]
    if not SEX:
        ax2.hist(data, n_bins, density=True, histtype='bar', label=['Ratio Average'])
        ax2.legend(prop={'size': 10})
        ax2.set_title('RSSI Ratio averages')
    else:
        fig.add_trace(go.Histogram(x=data, histnorm='percent', name='RSSI Ratio averages', nbinsx=n_bins), col=1, row=2)

    data_1 = [n for n in df['Samples above Old Ratio']]
    data_2 = [n for n in df['Samples below Old Ratio']]
    if not SEX:
        ax3.hist([data_1, data_2], n_bins, density=True, histtype='bar', label=['Samples above', 'Samples below'])
        ax3.legend(prop={'size': 10})
        ax3.set_title('KA Timeouts: Static Ratios')
    else:
        fig.add_trace(go.Histogram(x=[data_1, data_2], histnorm='percent', name='KA Timeouts: Static Ratios', nbinsx=n_bins), col=2, row=1)

    data_1 = [n for n in df['Samples above New Ratio']]
    data_2 = [n for n in df['Samples below New Ratio']]
    if not SEX:
        ax4.hist([data_1, data_2], n_bins, density=True, histtype='bar', label=['Samples above', 'Samples below'])
        ax4.legend(prop={'size': 10})
        ax4.set_title('KA Timeouts: Dynamic Ratios')
    else:
        fig.add_trace(go.Histogram(x=[data_1, data_2], histnorm='percent', name='KA Timeouts: Dynamic Ratios', nbinsx=n_bins), col=2, row=2)

    if not SEX:
        fig.tight_layout(pad=2)
        plt.savefig(file_out[:-5] + '.jpg', dpi=300)
    else:
        fig.update_layout(title_text='Optimizer RSSI Histogram', xaxis_title_text='Value', yaxis_title_text='Count', bargap=0.2, bargroupgap=0.1)
        fig.write_html(file_out, auto_open=auto_open_html)


if __name__ == "__main__":
    folder = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data"
    # folder = r"C:\Users\eddy.a\Downloads\Raw Data"
    # sub_folder = "100 days - (1.5, 0.6667) filter + 2 sample delay\\"
    # sub_folder = "100 days - no filter + 2 sample delay\\"
    sub_folder = "Plots\\"
    file_path = "100 days"
    file_path_output_csv = f"{folder}\\{sub_folder}{file_path} - analysis with {rssi_ratio_algorithm_enable[1]} Sample Delay.csv"

    # main_df = combine_dfs(f"{folder}\\{file_path} - RAW data.csv", f"{folder}\\P141.csv", file_path_output_csv)
    main_df = pd.read_csv(file_path_output_csv,  index_col=0).dropna(how='all', axis='columns')
    # plot_df(main_df, file_out=f"{folder}\\{sub_folder}Optimizer RSSI Monitor.html", auto_open_html=False)
    main_df = summarize(main_df)
    main_df.to_csv(f"{folder}\\{sub_folder}{file_path} - Summary.csv", index=False)
    # main_df = pd.read_csv(f"{folder}\\{sub_folder}{file_path} - Summary.csv").dropna(how='all', axis='columns')
    # main_df = pd.read_csv(f"{folder}\\{sub_folder}Summary.csv").dropna(how='all', axis='columns')
    plot_histogram(main_df, file_out=f"{folder}\\{sub_folder}Optimizer RSSI Histogram.html", auto_open_html=True, SEX=True)
    print()
