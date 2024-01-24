## Imports and main lists:
import os
import sys
import log_file
import plotly
import scipy.signal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime
from si_prefix import si_format
from plotly.subplots import make_subplots
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


## Variables:
filter_specific_file__name = [False, 'Spectrum 08 (12KΩ @ 5MHz - 20MHz).csv']
send_mail_when_finished__file__pc = [False, 'HFPLC - plot Spectrum with LTspice', 'eddyab-pc']
output_text = False
chrome_plot_auto_open__title = [True, 'Test 4.1 - 4x Optimizers (float)']
resample_ltspice = True
alpha_filter__alpha__times = [True, 0.1, 3]
find_peaks__min_f = [True, 16]
find_notches__min_f = [True, 16]
prominence__height__threshold = [2, None, None]
marker_dict = dict(size=8, color='rgb(255,0,0)', symbol='cross')
text_dict = dict(family='sans serif', size=15, color='#999')


## Paths and filters (use True or False):
filter_file_names = ['1KΩ', '12KΩ']
path_folder = r'M:\Users\Eddy A\Orion\03 D1836A - Test results\Test 1 - Optimizer\First test - All devices floating'
path_output = path_folder
include_subfolders = False
if not os.path.exists(path_output):
    os.makedirs(path_output)
if output_text:
    default_stdout = sys.stdout
# ## Spectrum:
spectrum_file_extension = '.csv'
spectrum_filter_file_name = 'Spectrum '
spectrum_delimiter = ','
spectrum_header_remove__lines__replace_with = [True, 45, ['Freq', 'Spectrum Data']]
spectrum_combine_files__only__to_csv = [True, True, True]
# ## LTspice:
ltspice_file_extension = '.txt'
ltspice_filter_file_name = ''
ltspice_header_remove__replace_with = [False, ['Freq', 'LTspice Data', 'del']]
ltspice_resample = True
ltspice_convert_to_dbm = [True, lambda x: 10 * np.log10((x / np.sqrt(2)) ** 2 / 0.05), lambda x: 10 ** (x / 20)]   # in Excel:   =10*LOG10((x/SQRT(2))^2/0.05)   =10^(x/20)



## Main:
print(f'main() - start. time = {datetime.now()}')
if output_text:
    sys.stdout = open(f'{path_output}/Python terminal Log ({datetime.now().strftime("%d-%m-%Y %H-%M-%S")}).txt', 'w')
    print(f'main() - start. time = {datetime.now()}')
for filter_file_name in filter_file_names:
    dfs_spectrum = []
    df_ltspice = None
    if include_subfolders:
        spectrum_file_names = [os.path.join(root, f) for root, dirs, files in os.walk(path_folder) for f in files if f.endswith(spectrum_file_extension) and f'{spectrum_filter_file_name}' in f]
        ltspice_file_names = [os.path.join(root, f) for root, dirs, files in os.walk(path_folder) for f in files if f.endswith(ltspice_file_extension) and f'{ltspice_filter_file_name}' in f]
    else:
        spectrum_file_names = [f for f in os.listdir(path_folder) if f.endswith(spectrum_file_extension) and f'{filter_file_name}' in f and f'{spectrum_filter_file_name}' in f]
        ltspice_file_names = [f for f in os.listdir(path_folder) if f.endswith(ltspice_file_extension) and f'{filter_file_name}' in f and f'{ltspice_filter_file_name}' in f]
    print(f'Got files; LTspice files = {len(ltspice_file_names)} and Spectrum files = {len(spectrum_file_names)}')

    for index_file, file in enumerate(ltspice_file_names):
        if filter_specific_file__name[0] and filter_specific_file__name[1] not in file:
            continue
        ltspice_file = open(path_folder + '\\' + file, "r").read()
        if 'dB' in ltspice_file:
            original_is_db = True
            ltspice_file = ltspice_file.replace('(', '')
            ltspice_file = ltspice_file.replace('dB', '')
            ltspice_file = ltspice_file.replace('°)', '')
        else:
            original_is_db = False
        ltspice_file = ltspice_file.replace('\t', ',')
        if not ltspice_header_remove__replace_with[0]:
            names = [s.title() for s in ltspice_file.split('\n')[0].replace('.', '').replace('V(', '').replace(')', ',del').replace('_', ' ').split(',')]
            names = [f'{s} {i}' if 'Del' in s else s for i, s in enumerate(names)]
            df_ltspice = pd.read_csv(StringIO(ltspice_file), delimiter=',', skiprows=1, names=names)
        else:
            df_ltspice = pd.read_csv(StringIO(ltspice_file), delimiter=',', skiprows=1, names=ltspice_header_remove__replace_with[1])
        df_ltspice = df_ltspice[df_ltspice.columns.drop(list(df_ltspice.filter(regex='Del')))]
        if ltspice_convert_to_dbm[0]:
            names = list(df_ltspice.head(0))[1:]
            if original_is_db:
                df_ltspice = df_ltspice.apply(lambda x: ltspice_convert_to_dbm[2](x) if x.name in names else x)
            df_ltspice = df_ltspice.apply(lambda x: ltspice_convert_to_dbm[1](x) if x.name in names else x)

    for index_file, file in enumerate(spectrum_file_names):
        if filter_specific_file__name[0] and filter_specific_file__name[1] not in file:
            continue
        if not spectrum_header_remove__lines__replace_with[0]:
            df = pd.read_csv(path_folder + '\\' + file, delimiter=spectrum_delimiter).dropna(how='all', axis='columns')
        else:
            df = pd.read_csv(path_folder + '\\' + file, delimiter=spectrum_delimiter, skiprows=spectrum_header_remove__lines__replace_with[1], names=spectrum_header_remove__lines__replace_with[2]).dropna(how='all', axis='columns')
        if alpha_filter__alpha__times[0] and alpha_filter__alpha__times[1] > 0:
            for i in range(alpha_filter__alpha__times[2]):
                df.iloc[:, 1] = log_file.alpha_beta_filter(df.iloc[:, 1], alpha_filter__alpha__times[1], True)
        dfs_spectrum.append(df)

    if len(dfs_spectrum) > 0:
        if spectrum_combine_files__only__to_csv[0]:
            if spectrum_combine_files__only__to_csv[1]:
                dfs_spectrum = [pd.concat(dfs_spectrum).drop_duplicates(subset=['Freq']).sort_values('Freq').reset_index(drop=True)]
            else:
                dfs_spectrum.append(pd.concat(dfs_spectrum).drop_duplicates(subset=['Freq']).sort_values('Freq').reset_index(drop=True))
            if spectrum_combine_files__only__to_csv[2]:
                dfs_spectrum[-1].to_csv(f'{path_output}/All df plots - {filter_file_name} {spectrum_filter_file_name}.csv', index=False, )
                df_ltspice.to_csv(f'{path_output}/LTspice - {filter_file_name} {spectrum_filter_file_name}.csv', index=False, )
                if spectrum_combine_files__only__to_csv[1]:
                    continue
        print('Level 1 finished')
        for i_df, df in enumerate(dfs_spectrum):
            fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
            traces = 0
            slider_steps = []
            fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], name=f'Spectrum {filter_file_name}', visible=True), col=1, row=1)
            traces += 1
            if find_peaks__min_f[0]:
                index_peaks, value_peaks = scipy.signal.find_peaks(df.iloc[:, 1][find_peaks__min_f[1]:], prominence=prominence__height__threshold[0], height=prominence__height__threshold[1], threshold=prominence__height__threshold[2])
                index_peaks = index_peaks + find_peaks__min_f[1]
                fig.add_trace(go.Scatter(x=[df.iloc[:, 0][j] for j in index_peaks], y=[df.iloc[:, 1][j] for j in index_peaks], mode='markers+text', text=[f'{df.iloc[:, 1][j]:.2f} @ {si_format(df.iloc[:, 0][j])}Hz' for j in index_peaks], textfont=text_dict, marker=marker_dict, name=f'Spectrum {filter_file_name} PEAKS', visible=True), col=1, row=1)
            if df_ltspice is not None:
                for index_df in range(1, df_ltspice.shape[1]):
                    if ltspice_resample:
                        x = list(df_ltspice.iloc[:, 0].squeeze())
                        y = df_ltspice.iloc[:, index_df].squeeze()
                        f = scipy.interpolate.interp1d(x, y)
                        x = list(df.iloc[:, 0].squeeze())
                        y = f(x)
                    fig.add_trace(go.Scatter(y=y, x=x, name=f'LTspice {filter_file_name} - {df_ltspice.keys()[index_df]}', visible=True), col=1, row=1)
                    traces += 1
                    if find_peaks__min_f[0]:
                        index_peaks, value_peaks = scipy.signal.find_peaks(y[find_peaks__min_f[1]:], prominence=prominence__height__threshold[0], height=prominence__height__threshold[1], threshold=prominence__height__threshold[2])
                        index_peaks = index_peaks + find_peaks__min_f[1]
                        fig.add_trace(go.Scatter(x=[x[j] for j in index_peaks], y=[y[j] for j in index_peaks], mode='markers+text', text=[f'{y[j]:.2f} @ {si_format(x[j])}Hz' for j in index_peaks], textfont=text_dict, marker=marker_dict, name=f'LTspice {filter_file_name} - {df_ltspice.keys()[index_df]} PEAKS', visible=True), col=1, row=1)
                        traces += 1
            if find_peaks__min_f[0]:
                skip_slider = 2
            else:
                skip_slider = 1
            slider_steps.append(dict(args=[{"visible": [True] * traces}, ], label='All visible'))
            for i_slider, slider in enumerate(range(2, traces, skip_slider)):
                step = dict(args=[{"visible": [True] * 2 + [False] * traces * 2}, ], label=f'Step {i_slider + 1} = {names[i_slider]}')
                step["args"][0]["visible"][slider] = True
                if find_peaks__min_f[0]:
                    step["args"][0]["visible"][slider + 1] = True
                slider_steps.append(step)
            slider_steps.append(dict(args=[{"visible": [True, False] * traces}, ], label='PEAKS off'))
            fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(xanchor="center", font=dict(size=16)))])
            if spectrum_combine_files__only__to_csv[0] and i_df == len(dfs_spectrum) - 1:
                file_out = chrome_plot_auto_open__title[1].split(' - ')[0] + f' - {filter_file_name} - ' + chrome_plot_auto_open__title[1].split(' - ')[1]
            else:
                file_out = f'{spectrum_file_names[i_df][:-4]} _ {chrome_plot_auto_open__title[1]}'
            fig.update_layout(title=file_out.split(' _ ')[0], title_font_color="#407294", title_font_size=40, legend_title="Plots:")
            plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}\\{file_out}.html', auto_open=chrome_plot_auto_open__title[0])

print(f'main() - finish. time = {datetime.now()}')
if output_text:
    sys.stdout.close()
    sys.stdout = default_stdout
    print(f'main() - finish. time = {datetime.now()}')
if send_mail_when_finished__file__pc[0]:
    _SM(pc=send_mail_when_finished__file__pc[2], file=send_mail_when_finished__file__pc[1])



# ## For when Level 1 finished
"""
dfs_spectrum = []
spectrum_file_names = []
file_out = 'LTspice(1kΩ)'
for i in range(1, df_ltspice.shape[1]):
    dfs_spectrum.append(df_ltspice.iloc[:, [0, i]])
    spectrum_file_names.append(list(df_ltspice.head(0))[i])


fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
traces = 0
slider_steps = []
for i_df, df in enumerate(dfs_spectrum):
    fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], name=f'{spectrum_file_names[i_df]}', visible=True), col=1, row=1)
    traces += 1
    if find_peaks__min_f[0]:
        index_peaks, value_peaks = scipy.signal.find_peaks(df.iloc[:, 1][find_peaks__min_f[1]:],
                                                           prominence=prominence__height__threshold[0],
                                                           height=prominence__height__threshold[1],
                                                           threshold=prominence__height__threshold[2])
        index_peaks = index_peaks + find_peaks__min_f[1]
        fig.add_trace(go.Scatter(x=[df.iloc[:, 0][j] for j in index_peaks], y=[df.iloc[:, 1][j] for j in index_peaks],
                                 mode='markers+text',
                                 text=[f'{df.iloc[:, 1][j]:.2f} @ {si_format(df.iloc[:, 0][j])}Hz' for j in index_peaks],
                                 textfont=text_dict, marker=marker_dict, name=f'{spectrum_file_names[i_df]} PEAKS',
                                 visible=True), col=1, row=1)


slider_steps = [dict(args=[{"visible": [True] * traces * 2}, ], label='All visible')]
skip_slider = 2
for i_slider, slider in enumerate(range(0, traces * 2, skip_slider)):
    step = dict(args=[{"visible": [False] * traces * 2}, ], label=f'Step {i_slider + 1} = {spectrum_file_names[i_slider]}')
    step["args"][0]["visible"][slider] = True
    if find_peaks__min_f[0]:
        step["args"][0]["visible"][slider + 1] = True
    slider_steps.append(step)
    print(f'{i_slider}')
slider_steps.append(dict(args=[{"visible": [True, False] * traces}, ], label='PEAKS off'))
fig.update_layout(sliders=[dict(active=0, pad={"t": 50}, steps=slider_steps, bgcolor="#ffb200", currentvalue=dict(xanchor="center", font=dict(size=16)))])
file_out = f'LTspice (1KΩ)'
if spectrum_combine_files__only__to_csv[0] and i_df == len(dfs_spectrum) - 1:
    file_out = chrome_plot_auto_open__title[1].split(' - ')[0] + f' - {filter_file_name} - ' + chrome_plot_auto_open__title[1].split(' - ')[1]
else:
    file_out = f'{spectrum_file_names[i_df][:-4]} _ {chrome_plot_auto_open__title[1]}'
fig.update_layout(title=file_out.split(' _ ')[0], title_font_color="#407294", title_font_size=40, legend_title="Plots:")
plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}\\{file_out}.html', auto_open=chrome_plot_auto_open__title[0])

"""
