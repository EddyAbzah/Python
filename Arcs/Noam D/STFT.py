import os
import gc
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import stft
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


auto_open_html = False


def read_csv(file_path, column=0):
    """Reads a CSV file and returns the selected column as a NumPy array."""
    df = pd.read_csv(file_path)
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    df.dropna(inplace=True)
    return pd.to_numeric(df.iloc[:, column], errors='coerce').values


def compute_stft(signal, fs=1.0, window_sec=0.256):
    """Computes the Short-Time Fourier Transform (STFT) with a window length in seconds."""
    nperseg = int(window_sec * fs)  # Convert seconds to samples
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # Convert magnitude to dB
    return f, t, Zxx_db


def plot_stft(f, t, Zxx_db, file_title="STFT"):
    """Plots the STFT with frequency on the x-axis and magnitude in dB on the y-axis, each window as a trace."""
    fig = go.Figure()
    for i, time in enumerate(t):
        fig.add_trace(go.Scatter(x=f, y=Zxx_db[:, i], mode='lines', name=f'Time {time:.2f}s'))

    fig.update_layout(
        title=file_title,
        xaxis_title='Frequency [Hz]',
        yaxis_title='Magnitude [dB]',
        legend_title='Time Windows'
    )
    return fig


def main():
    # file_in_path = r""
    folder_path = r""
    fs = 50e3 / 3
    window_sec = 0.5
    column = 0

    for file_in_path in [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("csv")]:
        print(f'{file_in_path = }')

        file_out_path = file_in_path[:-4] + " - STFT.html"
        file_title = file_out_path.rsplit("\\")[-1][:-5]

        signal = read_csv(file_in_path, column=column)
        f, t, Zxx_db = compute_stft(signal, fs, window_sec)
        fig = plot_stft(f, t, Zxx_db, file_title)
        plotly.offline.plot(fig, config={'scrollZoom': True, 'editable': True}, filename=file_out_path, auto_open=auto_open_html)

        del signal, f, t, Zxx_db
        gc.collect()


if __name__ == "__main__":
    main()
