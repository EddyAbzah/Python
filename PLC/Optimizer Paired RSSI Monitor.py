import os
import heartrate
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


# ## ### True ### ## # ## ### False ### ## #
# heartrate.trace(browser=True)

# ## DataFrame:
sorting_order = ['Portia ID', 'Optimizer ID', 'Date']
round_df_numbers = [True, {"Last RSSI": 0, "Paired RSSI": 0, "RSSI Ratio": 2, "Upper Ratio limit": 2, "Lower Ratio limit": 2}]
rename_df_columns = {"deviceid": "Portia ID", "managerid": "Manager ID", "time": "Date", "OPT ID": "Optimizer ID",
                     "param_129": "Last RSSI", "param_141": "Paired RSSI", "param_157": "Upper RSSI limit", "param_142": "Lower RSSI limit"}

# ## Algorythm:
rssi_ratio_algorythm_enable = [True, 1]
rssi_original_ratio = [2.51188643150958, 0.3981071705534972, 1]

# ## Plotting:
rssi_true__or__ratio_false = True
split_figs = 50


def rssi_ratio_algorythm(sdf):
    upper = [rssi_original_ratio[0]]
    lower = [rssi_original_ratio[1]]
    c_ratio = [rssi_original_ratio[2]]
    for index, sample in enumerate(sdf["RSSI Ratio"][:-1]):
        if upper[index - 1] < sample < lower[index - 1]:
            c = (sample + 1) / 2
            c_ratio.append(c)
            upper.append(upper[index - 1] * c)
            lower.append(lower[index - 1] * c)
        else:
            c_ratio.append(1)
            upper.append(upper[index - 1])
            lower.append(lower[index - 1])
    return pd.concat([sdf, pd.DataFrame({"New Upper Ratio limit": upper, "New Lower Ratio limit": lower, "New C Ratio": c_ratio}, index=sdf.index)], axis=1)


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
        df3['RSSI Ratio'] = df3["Last RSSI"] / df3["Paired RSSI"]
        df3["Old Upper Ratio limit"] = rssi_original_ratio[0]
        df3["Old Lower Ratio limit"] = rssi_original_ratio[1]
        df3["Old C Ratio"] = rssi_original_ratio[2]
        if not rssi_ratio_algorythm_enable[0]:
            df3["New Upper Ratio limit"] = rssi_original_ratio[0]
            df3["New Lower Ratio limit"] = rssi_original_ratio[1]
            df3["New C Ratio"] = rssi_original_ratio[2]
            df_full = df3
        else:
            df_full = pd.DataFrame()
            for optimizer_index, optimizer_id in enumerate(df3["Optimizer ID"].unique()):
                print(f'rssi_ratio_algorythm() for Optimizer number {optimizer_index + 1}: {optimizer_id = }: change ratio after {rssi_ratio_algorythm_enable[1]} KA timeouts)')
                result = rssi_ratio_algorythm(df3[df3["Optimizer ID"] == optimizer_id])
                df_full = pd.concat([df_full, result], axis=0)
        del df1
        del df2
        del df3
        if round_df_numbers[0]:
            df_full = df_full.round(round_df_numbers[1])
            for data_set, round_number in round_df_numbers[1].items():
                if round_number == 0:
                    df_full[data_set] = df_full[data_set].apply(int)
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
        print(f'Plotting Optimizer number {optimizer_index + 1}: {optimizer_id = }')
        sdf = df[df["Optimizer ID"] == optimizer_id]
        plot_title = f'Optimizer {optimizer_id} (Inverter {sdf["Portia ID"].iloc[0]:X} / {sdf["Manager ID"].iloc[0]:X}): KA timeouts = 0'
        if optimizer_index % split_figs == 0:
            fig = make_subplots(cols=1, rows=1, shared_xaxes=False)
            steps = list()
            fig_count += 1

        if rssi_true__or__ratio_false:
            for trace in ["Last RSSI", "Paired RSSI", "Old C Ratio"]:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace], name=f"{optimizer_id} - {trace}", visible=optimizer_index % split_figs == 0,
                                         hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace + ': %{y}<extra></extra>'), col=1, row=1)
            for trace1, trace2 in [("Old Upper Ratio limit", "Paired RSSI"), ("Old Lower Ratio limit", "Paired RSSI")]:
                fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace1] * sdf[trace2], name=f"{optimizer_id} - {trace1}", visible=optimizer_index % split_figs == 0,
                                         hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace1 + ': %{y}<extra></extra>'), col=1, row=1)
            if rssi_ratio_algorythm_enable[0]:
                for trace1, trace2 in [("New Upper Ratio limit", "Paired RSSI"), ("New Lower Ratio limit", "Paired RSSI")]:
                    fig.add_trace(go.Scatter(x=list(sdf.index), y=sdf[trace1] * sdf[trace2], name=f"{optimizer_id} - {trace1}", visible=optimizer_index % split_figs == 0,
                                             hovertemplate=f'Optimizer HEX ID: {optimizer_id}<br>' + 'Date: %{x}<br>' + trace1 + ': %{y}<extra></extra>'), col=1, row=1)
        else:
            ...
        if optimizer_index % split_figs == 0:
            figs_per_fig = len(fig.data)
            print(f'{figs_per_fig = }')
        step = dict(label=f'Optimizer HEX ID: {optimizer_id}', method="update",
                    args=[{"visible": [False] * figs_per_fig * split_figs}, {"title": plot_title}])
        step["args"][0]["visible"][optimizer_index % split_figs * figs_per_fig:optimizer_index % split_figs * figs_per_fig + figs_per_fig] = [True] * figs_per_fig
        steps.append(step)

        if (optimizer_index + 1) % split_figs == 0 or optimizer_index == df["Optimizer ID"].nunique() - 1:
            fig.update_layout(title=plot_title, title_font_color="#2589BB", title_font_size=40, legend_title="Traces:", legend_title_font_color="#2589BB")
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=steps)])
            fig.write_html(f"{file_out[:-5]} {fig_count:02} - Optimizers {optimizer_index - split_figs + 1} to {optimizer_index + 1}.{file_out[-4:]}", auto_open=auto_open_html)

            break


if __name__ == "__main__":
    # folder = r"M:\Users\ShacharB\Projects\PLC Leakage - RSSI Ratio Issue\RSSI Ratio Issue Gen4 - 12.2022\Solution Procedure\Raw Data"
    folder = r"C:\Users\eddy.a\Downloads\RSSI Ratio\Raw Data"
    file_path_in1 = folder + "\\" + "Raw data 40 days.csv"
    file_path_in2 = folder + "\\" + "P141.csv"
    file_path_output_csv = folder + "\\" + "Raw data 40 days (edited).csv"
    file_path_output_html = folder + "\\" + "MANA.html"

    df = combine_dfs(file_path_in1, file_path_in2, file_path_output_csv)
    # df = pd.read_csv(file_path_output_csv,  index_col=0).dropna(how='all', axis='columns')
    plot_df(df, file_out=file_path_output_html, auto_open_html=True)
