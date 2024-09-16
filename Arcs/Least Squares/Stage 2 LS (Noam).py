import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import signal
from numpy.linalg import inv
from plotly.subplots import make_subplots

# ####   True   ###   False   #### #
folder = r'V:\HW_Infrastructure\Analog_Team\ARC_Data\Results\Jupiter\Jupiter+ Improved (7E0872F4-EC)\Stage 2 Validation (27-08-2023)' + '\\'
# folder += 'Stage 2 Validation 01 - failed (21-08-2023)'
# folder += 'Stage 2 Validation 02 - Single String KA1 (21-08-2023)'
# folder += 'Stage 2 Validation 03 - Two Strings KA1 (21-08-2023)'
folder += 'Stage 2 Validation 04 - KA1 + KA2 (22-08-2023)'
# folder += 'Stage 2 Validation 05 - KA2 (23-08-2023)'
file_filter_1 = ['spi rec', '.txt']     # to include
file_filter_2 = ['002', '003']          # to exclude
title_df_filter = ['rxout', 'rx out', 'adc in', 'spi sample']
plot_cut_before_spi = [True, 1, -2100, 17600]        # 0 = ToF, 1 = State Machine, 2 = start trim, 3 = desired length
output_name_list = ['SPI Sample Least Squares', 'Least Squares Self Test 300 samples']
decimate_scope_input = [False, 4]   # from Fs = 50kHz to 16.667kHz or 12.5kHz
output_zero_first_calc = [True, 2]
iir_filter__alpha = [False, 0.5]
avg_count__add__cut = [5, 2, 1]
auto_open_html = False
new_v3_method = True
N = 100
fs = 16667
PLL_freq = [511.402344, 492.738281]
self_test_freq__phase__amp_shifs = [False, PLL_freq[0], [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180] * 4, [1] * 13 + [0.75] * 13 + [0.5] * 13 + [0.25] * 13]


def main():
    files = sorted([f for f in os.listdir(folder) if all(s in f.lower() for s in file_filter_1) and any(s in f.lower() for s in file_filter_2)])
    print("File name\tDiff Phase shift\tDiff Amplitude Ratio\tDiff indexes\tMin Phase shift\tMax Phase shift\tMin Amplitude Ratio\tMax Amplitude Ratio\tMin Max indexes")
    if len(files) > 0 or self_test_freq__phase__amp_shifs[0]:
        Ls_phase_est(folder, files)
    else:
        print('files is empty')


def LS(H, y):
    H = np.mat(H)
    HtH = np.matmul(H.T, H)
    inv_HtH = inv(HtH)
    inv_HtH_Ht = np.matmul(inv_HtH, H.T)
    inv_HtH_Ht_y = np.matmul(inv_HtH_Ht, y)
    para = inv_HtH_Ht_y
    return (inv_HtH_Ht_y)


def generate_sine_wave(freq, sample_freq, phase_jump, amp_jump, num_cycles):
    T = 1.0 / sample_freq
    num_points = int(num_cycles / freq * sample_freq)
    t = np.linspace(0, num_points * T, num_points, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)
    sine_wave[int(num_points / 2):] = np.sin(2 * np.pi * freq * t[int(num_points / 2):] + np.deg2rad(phase_jump))   # Apply phase jump at the middle
    sine_wave[int(num_points / 2):] = sine_wave[int(num_points / 2):] * amp_jump   # Apply amp jump at the middle
    return t, sine_wave


def Ls_phase_est(folder, files):
    row = 3
    col = 1
    plots_per_pane = 5
    fig = initialize_fig(row=row, col=col, plots_per_pane=plots_per_pane, shared_xaxes=True, subplot_titles=['Signal', 'Phase', 'Amplitude'])
    if self_test_freq__phase__amp_shifs[0]:
        files = []
        for phase, amp in zip(self_test_freq__phase__amp_shifs[2], self_test_freq__phase__amp_shifs[3]):
            files.append(f'phase shift = {phase} and amp ratio = {amp}')
        output_name = output_name_list[1]
    else:
        output_name = output_name_list[0]

    for file_index, file in enumerate(files):
        if self_test_freq__phase__amp_shifs[0]:
            w = 2 * np.pi * (self_test_freq__phase__amp_shifs[1] / fs)
        elif 'ka2' in file.lower():
            w = 2 * np.pi * (PLL_freq[1] / fs)
        else:
            w = 2 * np.pi * (PLL_freq[0] / fs)

        if self_test_freq__phase__amp_shifs[0]:
            phase = self_test_freq__phase__amp_shifs[2][file_index]
            amp = self_test_freq__phase__amp_shifs[3][file_index]
            time, sine_wave = generate_sine_wave(self_test_freq__phase__amp_shifs[1], fs, phase, amp, 300)
            input = sine_wave.flatten()
            input_time = time.flatten()
            y = input
        else:
            dfs = pd.read_csv(folder + '\\' + file).dropna(how='all', axis='columns')
            elif plot_cut_before_spi[0]:
                indexes = dfs[dfs['Machine State'].diff() == plot_cut_before_spi[1]].index.tolist()
                if indexes is None or len(indexes) == 0:
                    print(f'The porper State Machine (={plot_cut_before_spi[1]}) was not found... halving the record')
                    indexes = [int(len(dfs) / 2), int(len(dfs) / 2) + plot_cut_before_spi[3]]
                elif len(indexes) == 1:
                    indexes.append(indexes[0] + plot_cut_before_spi[3])
                else:
                    indexes = indexes[-2:]
                dfs = dfs[indexes[0] + plot_cut_before_spi[2]:indexes[1] + plot_cut_before_spi[2]].reset_index(drop=True)
            input = dfs['SPI Sample'].to_numpy().flatten()
            input_time = np.arange(0, len(input.flatten())) / fs
            y = input.flatten()
        y_sliced = y[0:(len(y) // N) * N]
        y = np.reshape(y_sliced, ((int(len(y_sliced) / N)), N))
        y = y.T

        # init matrix H
        H = np.zeros((N, 2))
        Jump = 0

        para_x, para_y = [], []
        for z in range(0, len(y_sliced) // N):
            for k in range(0, N):
                H[k] = [np.sin(w * ((k) + Jump)), np.cos(w * ((k) + Jump))]  # building the row of the matrix.
            Jump += N
            Vec1 = y.T[z]
            para = LS(H, Vec1)
            para_y.append(para[0, 1])
            para_x.append(para[0, 0])

        para_x = np.array(para_x)
        para_y = np.array(para_y)
        para_x_g = para_x[1:1000]
        para_y_g = para_y[1:1000]

        comlex_from_para_g = (para_x_g + 1j * para_y_g).T;
        ### Second LS
        H2 = np.array([comlex_from_para_g[1:]]).T
        Vec2 = np.array(comlex_from_para_g[:-1].T)
        Filter_exp_val = LS(H2, Vec2)
        complex_from_para = (para_x + 1j * para_y)

        # filter = [1, -Filter_exp_val.H]
        # filter_applied = np.convolve(filter, comlex_from_para)

        exp_fix = (Filter_exp_val.getA() ** np.array(range(len(complex_from_para))))[0]  # illuminate delta omega
        phase_jump = exp_fix * complex_from_para
        abs_phase = np.arctan2(np.imag(phase_jump), np.real(phase_jump))
        phase_jump_no_transient_effect = phase_jump[10:] / phase_jump[:-10]
        if new_v3_method:
            phase_unwarp = np.arctan2(np.imag(phase_jump_no_transient_effect), np.real(phase_jump_no_transient_effect));
        else:
            phase_unwarp = np.arctan(np.imag(phase_jump_no_transient_effect) / np.real(phase_jump_no_transient_effect))
        Amp = abs(phase_jump_no_transient_effect)
        phase_time = np.arange(0, len(phase_unwarp)) / (fs / N)
        phase_unwarp = phase_unwarp * 180 / np.pi
        fig.add_trace(go.Scattergl(x=input_time, y=input[:], name=f'input', mode="lines", visible=False, line=dict(color="red"), showlegend=True), row=1, col=1)
        if iir_filter__alpha[0]:
            Amp = signal.lfilter([iir_filter__alpha[1]], [1, -iir_filter__alpha[1]], Amp)
            phase_unwarp = signal.lfilter([iir_filter__alpha[1]], [1, -iir_filter__alpha[1]], phase_unwarp)


        if output_zero_first_calc[0]:
            temp_i = output_zero_first_calc[1] + 1
            phase_unwarp[:temp_i] = [phase_unwarp[temp_i]] * temp_i
            Amp[:temp_i] = [Amp[temp_i]] * temp_i
        indexes = list(np.argpartition(abs(np.nan_to_num(np.diff(phase_unwarp))), -2)[-2:])
        indexes.extend(np.argpartition(abs(np.nan_to_num(np.diff(phase_unwarp))), -2)[-2:])
        indexes = np.sort(indexes)[1:3]
        while indexes[1] - indexes[0] > avg_count__add__cut[0]:
            indexes[0] += avg_count__add__cut[1]
            indexes[1] -= avg_count__add__cut[2]
        if len(phase_unwarp[indexes[0]:indexes[1]]) > 0:
            phase_shift = phase_unwarp[indexes[0]:indexes[1]].mean()
            amp_ratio = Amp[indexes[0]:indexes[1]].mean()
            xxx = phase_time[indexes[0]:indexes[1]]
            yyy1 = phase_unwarp[indexes[0]: indexes[1]]
            yyy2 = Amp[indexes[0]: indexes[1]]
        else:
            phase_shift = 0
            amp_ratio = 1
            xxx = [0]
            yyy1 = [0]
            yyy2 = [0]
        min_max = f'{phase_unwarp.max():.2f}\t{phase_unwarp.min():.2f}\t{Amp.max():.2f}\t{Amp.min():.2f}\t{phase_unwarp.argmax()}, {phase_unwarp.argmin()}, {Amp.argmax()}, {Amp.argmin()}'

        fig.add_trace(go.Scattergl(x=phase_time, y=phase_unwarp, name='Phase [°]', mode="lines", visible=False, line=dict(color="blue"), showlegend=True), row=2, col=1)
        fig.add_trace(go.Scatter(mode='lines+markers', name=f'Phase shift = {phase_shift:.2f}', visible=False, line=dict(color="blue"), x=xxx, y=yyy1), row=2, col=1)
        fig.add_trace(go.Scattergl(x=phase_time, y=Amp, name='Amplitude', mode="lines", visible=False, line=dict(color="orange"), showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(mode='lines+markers', name=f'Amp ratio = {amp_ratio:.2f}', visible=False, line=dict(color="orange"), x=xxx, y=yyy2), row=3, col=1)
        print(f'{file}\t{phase_shift:.2f}\t{amp_ratio:.2f}\t{indexes[0]} - {indexes[1]}\t{min_max}')

    for i in range(plots_per_pane):
        fig.data[i].visible = True
    steps = []
    col_names_list = files
    for i in range(0, int(len(fig.data) / plots_per_pane)):
        Temp = col_names_list[i]
        if not self_test_freq__phase__amp_shifs[0]:
            Temp = Temp[:Temp.find('Rec') + 6]
        step = dict(method="update", args=[{"visible": [False] * len(fig.data)}, {"title": f"{output_name}: {str(Temp)}"}], label=str(Temp))
        j = i * plots_per_pane
        for k in range(plots_per_pane):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict( active=10, currentvalue={"prefix": "REC: "}, pad={"t": 50}, steps=steps)]
    fig.update_layout(title=output_name, sliders=sliders)
    write_html_file = f'{folder}\\{output_name}.html'
    if auto_open_html:
        fig.write_html(write_html_file, config={'scrollZoom': True, 'responsive': False}, auto_open=auto_open_html)


def initialize_fig(row=4, col=1, plots_per_pane=4, shared_xaxes=True, subplot_titles=['Rx', 'Rx', 'Rx', 'Rx']):
    all_specs = np.array([[{"secondary_y": True}] for x in range((row * col))])
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()
    fig = make_subplots(rows=row, cols=col, specs=all_specs_reshaped, shared_xaxes=shared_xaxes, subplot_titles=subplot_titles)
    return fig


if __name__ == "__main__":
    main()
