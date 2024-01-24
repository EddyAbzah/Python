import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import signal
from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
import Plot_Graphs_with_Sliders as _G
import my_tools


def butterworth_2nd_filter(freq, log, Av, res_1, res_2, cap_1, cap_2):
    """-40 dB/Decade - no resonance"""
    fc = 1 / (2 * np.pi * np.sqrt(res_1 * res_2 * cap_1 * cap_2))
    mana = log.copy()
    for index, f in enumerate(freq):
        if Av != 1:
            log[index] = log[index] * Av
        if f > fc:
            log[index] = 0
    return log


def main():
    path_output = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
    path_file_folder = r'M:\Users\HW Infrastructure\PLC team\ARC\Temp-Eddy'
    path_file_name = 'Spectrum 1kHz-260kHz'
    path_file_filter = ['', ' with SS and BB']
    path_file_filter = [' - Only TX1 (Max Hold)', ' - TX1-RX1 (Max Hold)']
    chrome_auto_open = True
    Mixer_LO = [80]
    # Mixer_LO = [i * 10 for i in range(2, 31)]
    No_Harmonics = 3
    resample_n = 3000

    for mixer in Mixer_LO:
        path_file_output = f'Mixer {mixer}kHz (Harmonics = {No_Harmonics})'
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        for s in path_file_filter:
            df = pd.read_csv(f'{path_file_folder}/{path_file_name+s}.csv')
            # x_axis = df.pop(list(df.head(0))[0])
            plot_main = df[[f for f in df.head(0) if 'Max Hold' in f]].max(axis=1)
            f = np.linspace(1000, 260e3, resample_n)[4:]
            Actual_RF_frequency = signal.resample(plot_main.to_numpy().flatten(), resample_n)[4:]
            butterworth_2nd_filter(f, Actual_RF_frequency, 1, 649, 649, 3.9e-9, 1e-9)    # Diff Amp LPF
            plot_main_after_foldings = multiplicationTable_V2(Actual_RF_frequency, No_Harmonics, mixer * 1000, f)
            fig.add_trace(go.Scatter(x=f, y=Actual_RF_frequency, name=f'Original{s}', showlegend=True), col=1, row=1)
            fig.add_trace(go.Scatter(x=f, y=plot_main_after_foldings, name=f'Folded{s}', showlegend=True), col=1, row=1)
        fig.update_layout(title=path_file_output, title_font_color="#407294", title_font_size=40,
                          legend_title="Plots:", legend_title_font_color="green")
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=f'{path_output}/{path_file_output}.html',
                            auto_open=chrome_auto_open)
    print('FINISHED')


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]


def return_harmonic(val, mixer_harmonic, rf_harmonic):
    return val - (mixer_harmonic + rf_harmonic) * 15


def multiplicationTable_V2(Actual_RF_frequency, No_Harmonics, Mixer_LO, f):
    if not isinstance(Mixer_LO, list):
        Mixer_LO = [Mixer_LO]
    index_RF = np.arange(1, No_Harmonics + 1)
    index_LO = np.arange(1, No_Harmonics + 1)
    array_copy = -100 * np.ones(len(Actual_RF_frequency))
    zipped_rf_list = list(zip(f, Actual_RF_frequency))

    for one_rf in zipped_rf_list:
        ff = one_rf[0]
        val = one_rf[1]
        for Mixer_LO_index, Mixer_LO_value in enumerate(Mixer_LO):
            for LO_index, LO_index_value in enumerate(index_LO):
                for RF_index, RF_index_value in enumerate(index_RF):
                    temp_freq_SUB = abs(RF_index_value * ff - LO_index_value * Mixer_LO_value)
                    index_SUB = np.where(f == find_nearest(f, temp_freq_SUB))
                    val = return_harmonic(val, LO_index, RF_index)

                    if array_copy[index_SUB] < val:
                        print('index sub is ' + str(index_SUB))
                        print('Old value is' + str(array_copy[index_SUB]))
                        print('New value is ' + str(val))
                        array_copy[index_SUB] = val

                    temp_freq_ADD = abs(RF_index_value * ff + LO_index_value * Mixer_LO_value)
                    index_ADD = np.where(f == find_nearest(f, temp_freq_ADD))
                    if array_copy[index_ADD] < val:
                        print('index add is ' + str(index_ADD))
                        print('Old value is' + str(array_copy[index_ADD]))
                        print('New value is ' + str(val))
                        array_copy[index_ADD] = val

    return array_copy


if __name__ == "__main__":
    main()
