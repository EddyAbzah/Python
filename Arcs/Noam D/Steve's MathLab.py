import os
import gc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import welch
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


def load_data(filename):
    """ Load CSV file data """
    try:
        dat_raw = pd.read_csv(filename)
        dat_raw = dat_raw[column]
        return dat_raw
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")


def compute_psd(dat):
    """ Compute Power Spectral Density """
    f, Pxx = welch(dat, Fs, nperseg=1024, noverlap=768)
    return f, np.sqrt(Pxx)


def energy_calculation(dat):
    blp = 25 / 46 - 21 / 46 * np.cos(2 * np.pi * np.arange(49) / 48)
    blp /= np.sum(blp)
    t_filt = np.arange(len(blp)) / Fs
    blpzs = np.sin(2 * np.pi * f0 * t_filt) * blp
    blpzc = np.cos(2 * np.pi * f0 * t_filt) * blp
    blpos = np.sin(2 * np.pi * f1 * t_filt) * blp
    blpoc = np.cos(2 * np.pi * f1 * t_filt) * blp

    zz = np.convolve(dat, blpzs, mode='same') ** 2 + np.convolve(dat, blpzc, mode='same') ** 2
    oo = np.convolve(dat, blpos, mode='same') ** 2 + np.convolve(dat, blpoc, mode='same') ** 2
    return zz, oo


def detect_bits(oo, zz):
    return np.where(oo > zz, 1, 0)


if __name__ == '__main__':
    Fs = 50e3 / 3  # Sampling frequency
    Rsym = 679.8096533  # Symbol rate
    f0 = 1505.789  # Frequency 1
    f1 = 2852.592  # Frequency 2
    column = "RX1(1)"
    auto_open_html = False
    folder_path = r"C:\Users\eddy.a\Downloads\Noise Floor Test 03\14 SPI RX1"

    for filename in [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("csv")]:
        dat = load_data(filename)
        t = np.arange(len(dat)) / Fs

        # Preprocess signal
        dat = dat - np.mean(dat)
        dat = dat * 2.5 / 16384

        # Compute PSDs
        f_orig, Pxx_orig = compute_psd(dat)
        Signal, _ = dat, None  # DAGC is not modified in this step
        f_dagc, Pxx_dagc = compute_psd(Signal)

        zz, oo = energy_calculation(dat)
        zdbm = 10 * np.log10(zz / 50) + 30
        odbm = 10 * np.log10(oo / 50) + 30

        bs = detect_bits(oo, zz)

        # Compute Bit-change timing histogram
        bit_change_indices = np.where(np.diff(bs) != 0)[0]
        bit_diffs = np.diff(bit_change_indices)
        fig_histogram = go.Figure()
        fig_histogram.add_trace(go.Histogram(x=bit_diffs, nbinsx=500, name="Bit-Change Timing"))

        # Create interactive figure with slider
        fig = go.Figure()

        # Add traces (only the first one is visible initially)
        fig.add_trace(go.Scatter(x=t, y=Signal, mode='lines', name='Original PSD', visible=True))
        fig.add_trace(go.Scatter(x=f_dagc, y=Pxx_dagc, mode='lines', name='DAGC PSD', visible=False))

        # Add Zeros & Ones traces (initially hidden)
        fig.add_trace(go.Scatter(x=t, y=zdbm, mode='lines', name='Zeros Energy', visible=False))
        fig.add_trace(go.Scatter(x=t, y=odbm, mode='lines', name='Ones Energy', visible=False))

        # Add Bit-Change Histogram (initially hidden)
        fig.add_trace(go.Histogram(x=bit_diffs, nbinsx=500, name="Bit-Change Timing", visible=False))

        # Create slider steps
        steps = []

        # Step for Original PSD
        steps.append(dict(
            method="update",
            args=[
                {"visible": [True, False, False, False, False]},  # Show only Original PSD
                {"title": "Power Spectral Density - Original Signal", "xaxis": {"title": "Frequency [Hz]"}, "yaxis": {"title": "Amplitude"}}
            ],
            label="Original PSD"
        ))

        # Step for DAGC PSD
        steps.append(dict(
            method="update",
            args=[
                {"visible": [False, True, False, False, False]},  # Show only DAGC PSD
                {"title": "Power Spectral Density - DAGC Signal", "xaxis": {"title": "Frequency [Hz]"}, "yaxis": {"title": "Amplitude"}}
            ],
            label="DAGC PSD"
        ))

        # Step for Zeros & Ones Energy (both traces visible together)
        steps.append(dict(
            method="update",
            args=[
                {"visible": [False, False, True, True, False]},  # Show both Zero & One Energy traces
                {"title": "Energy of Zeros and Ones in dBm", "xaxis": {"title": "Time [s]"}, "yaxis": {"title": "Energy (dBm)"}}
            ],
            label="Zeros & Ones Energy"
        ))

        # Step for Bit-Change Timing Histogram
        steps.append(dict(
            method="update",
            args=[
                {"visible": [False, False, False, False, True]},  # Show only Bit-Change Timing Histogram
                {"title": "Bit-Change Timing Histogram", "xaxis": {"title": "Bit Change Interval"}, "yaxis": {"title": "Count"}}
            ],
            label="Bit-Change Histogram"
        ))

        # Add slider to layout
        fig.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Current View: "},
                pad={"t": 50},
                steps=steps
            )],
            title="Power Spectral Density Analysis",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Amplitude"
        )

        fig.write_html(filename[:-4] + " - Steve's MathLab.html", config={"editable": True}, auto_open=auto_open_html)
        gc.collect()
