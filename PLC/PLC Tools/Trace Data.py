import os
import numpy as np
import pandas as pd
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


number_of_harmonics = 5
dc_cut = True
plot_each_fft = False


def analyze_fft(df, folder, filename):
    """
    Gets a pd.df of a Scope measurements, and return the amplitudes of the first 5 harmonics.
    Needs "Time" column to work.
    Frequency in [Hz], and Amplitude in [V].
    """
    summary_data = []

    if 'Time' not in df.columns:
        print(f"Column 'Time' not found in {filename}. Skipping.")
        return summary_data
    elif len(df) == 0:
        print(f"DataFrame {filename} is empty. Filling all with 'N/A'.")
        for column_name in df.columns:
            if column_name == 'Time':
                continue
            summary_row = [folder, filename, column_name, "N/A"]
            for _ in range(0, number_of_harmonics):
                summary_row.extend(["N/A", "N/A"])
            summary_data.append(summary_row)
        return summary_data

    time = df['Time'].values
    sampling_rate = 1 / np.mean(np.diff(time))
    for column_name in df.columns:
        if column_name == 'Time':
            continue
        signal = df[column_name].values

        # Compute FFT
        n = len(signal)
        fft_values = fft(signal)
        fft_amplitudes = 2 / n * np.abs(fft_values[:n // 2])
        frequencies = fftfreq(n, d=1 / sampling_rate)[:n // 2]

        if plot_each_fft:
            plt.figure()
            plt.plot(frequencies, fft_amplitudes)
            plt.title(f"FFT of {column_name} in {filename}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.show()

        # Extract the fundamental frequency and first 5 harmonics (including)
        if dc_cut:
            fundamental_index = np.argmax(fft_amplitudes[1:]) + 1
        else:
            fundamental_index = np.argmax(fft_amplitudes)
        fundamental_freq = frequencies[fundamental_index]
        fundamental_amp = fft_amplitudes[fundamental_index]
        harmonics = [(fundamental_freq, fundamental_amp)]
        for i in range(2, 1 + number_of_harmonics):  # Start from the 2nd harmonic
            harmonic_index = np.argmin(np.abs(frequencies - i * fundamental_freq))
            harmonics.append((frequencies[harmonic_index], fft_amplitudes[harmonic_index]))

        # Append results to the summary table
        summary_row = [folder, filename, column_name, np.sqrt(np.mean(signal**2))]      #File name, Measurement, Vrms
        for freq, amp in harmonics:
            summary_row.extend([freq, amp])
        summary_data.append(summary_row)

    return summary_data


if __name__ == '__main__':
    path = r"M:\Users\HW Infrastructure\PLC team\INVs\Venus4\Tests Results\V4 RevC RX Test - Eddy 11.2024\RX Tests"
    search_subfolders = True
    filter_in = ["*scop*data*.csv"]
    filter_out = ["spectrum"]
    csv_out = "_Scope Measurements.csv"
    rename = [True, {"Lrx": "01 Lrx", "DiffOut": "02 Diff Out", "BPFPLC": "03 BPF PLC", "BPFArc": "04 BPF Arc",
                     "LPF1": "05 LPF1", "Mixer1": "06 Mixer1", "Gain1": "07 Gain1",
                     "LPF2": "08 LPF2", "Mixer2": "09 Mixer2", "Gain2": "10 Gain2",
                     "LPF3": "11 LPF3", "Mixer3": "12 Mixer3", "Gain3": "13 Gain3",
                     "LPF4": "14 LPF4", "Mixer4": "15 Mixer4", "Gain4": "16 Gain4"}]

    summary_data = []
    file_index = 0
    for folder, subfolders, file_names in os.walk(path):
        for filename in file_names:
            if (filter_in[0] == "" or any(fnmatch(filename, f) for f in filter_in)) and (filter_out[0] == "" or not any(fnmatch(filename, f) for f in filter_out)):
                file_index += 1
                print(f'Analyzing file number {file_index:03}: {filename}')
                df = pd.read_csv(os.path.join(folder, filename))
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                # _P(df.set_index("Frequency"))
                summary_out = analyze_fft(df, folder, filename)
                summary_data.extend(summary_out)
        if not search_subfolders:
            break

    columns = ['Folder path', 'File Name', 'Trace Name', 'Vrms']
    for i in range(1, number_of_harmonics + 1):
        columns.extend([f'Harmonic {i} Frequency', f'Harmonic {i} Amplitude'])
    summary = pd.DataFrame(summary_data, columns=columns)
    if rename[0]:
        summary = summary.replace(rename[1])
    if csv_out != "":
        summary.to_csv(os.path.join(path, csv_out), index=False)
    print(summary.to_string())
