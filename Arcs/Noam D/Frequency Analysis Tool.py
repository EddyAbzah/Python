import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.signal import stft
import webbrowser
import threading
# Function to plot SPI files
def plot_spi_files(folder_path, files, fs=50e3 / 3, progress_text=None, Secondy_axis=False):
    # Get time and date and save plot_time_date
    import datetime
    now = datetime.datetime.now()
    time_date = now.strftime("%Y-%m-%d")
    plots_folder = os.path.join(folder_path, f'plots_{time_date}')
    os.makedirs(plots_folder, exist_ok=True)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if progress_text:
            progress_text.insert(tk.END, f"Processing file: {file_name}\n")
            progress_text.see(tk.END)
            progress_text.update()

        try:
            if file_name.endswith('.txt'):
                data = pd.read_csv(file_path)
            elif file_name.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file_path)
            else:
                continue

            if data.empty or len(data.columns) < 2:
                if progress_text:
                    progress_text.insert(tk.END, f"Skipping file {file_name}: Not enough data to plot.\n")
                    progress_text.see(tk.END)
                    progress_text.update()
                continue

            num_points = len(data)
            time = [i / fs for i in range(num_points)]

            primary_columns = []
            secondary_columns = []
            for col in data.columns:
                if Secondy_axis and data[col].max() > data[primary_columns[0]].max() if primary_columns else 0:
                    secondary_columns.append(col)
                    print('file_name:', file_name)
                    print(f"Secondary Axis: {col}")
                else:
                    primary_columns.append(col)

            fig = go.Figure()
            for column in data.columns:
                if column in secondary_columns:
                    fig.add_trace(go.Scatter(x=time, y=data[column], mode='lines', name=f"{column} (Secondary)", yaxis="y2"))
                else:
                    fig.add_trace(go.Scatter(x=time, y=data[column], mode='lines', name=f"{column} (Primary)"))

            fig.update_layout(
                title=f"Plot for {file_name}",
                xaxis_title="Time (s)",
                yaxis_title="Primary Axis Values",
                yaxis2=dict(title="Secondary Axis Values", overlaying="y", side="right"),
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_white",
            )
            config = dict({'scrollZoom': True})
            fig.update_layout(dragmode="zoom")

            plot_file_path = os.path.join(plots_folder, f"{os.path.splitext(file_name)[0]}.html")
            fig.write_html(plot_file_path, auto_open=False, config=config)

            if progress_text:
                progress_text.insert(tk.END, f"Saved plot to ")
                progress_text.insert(tk.END, "\n")
                progress_text.insert(tk.END, plot_file_path, ("link", plot_file_path))
                progress_text.insert(tk.END, "\n")
                progress_text.insert(tk.END, "------------------------------------------------------------\n", "red_line")
                progress_text.see(tk.END)
                progress_text.update()

                # Add tag to make the link clickable
                progress_text.tag_config("link", foreground="blue", underline=True)
                progress_text.tag_bind("link", "<Button-1>", lambda e, path=plot_file_path: webbrowser.open(path))
                progress_text.tag_config("red_line", foreground="red")
        except Exception as e:
            if progress_text:
                progress_text.insert(tk.END, f"Error processing file {file_name}: {e}\n")
                progress_text.see(tk.END)
                progress_text.update()
def original_tool():
    def preprocess_data(df, sampling_rate, downsample_factor=1):
        if downsample_factor > 1:
            df = df.iloc[::downsample_factor].reset_index(drop=True)
            sampling_rate = sampling_rate / downsample_factor

        if 'Time' not in df.columns:
            df['Time'] = np.arange(len(df)) / sampling_rate

        return df, sampling_rate

    def compute_stft(data, sampling_rate, f_resolution, t_resolution, overlap=True):
        nperseg = int(sampling_rate / f_resolution)
        step = int(t_resolution * sampling_rate)
        noverlap = nperseg - step if overlap else 0

        if noverlap < 0:
            raise ValueError("Overlap cannot be negative; check time and frequency resolutions.")

        f, t, Zxx = stft(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, window='hann')
        return f, t, Zxx

    def compute_goertzel(data, sampling_rate, target_freqs, window_size):
        num_windows = len(data) // window_size
        results = {freq: [] for freq in target_freqs}

        for w in range(num_windows):
            window_data = data[w * window_size:(w + 1) * window_size]
            N = len(window_data)

            for freq in target_freqs:
                k = int(0.5 + (N * freq) / sampling_rate)
                w = (2 * np.pi * k) / N
                coeff = 2 * np.cos(w)

                Q_prev = 0
                Q_prev2 = 0
                for sample in window_data:
                    Q = coeff * Q_prev - Q_prev2 + sample
                    Q_prev2 = Q_prev
                    Q_prev = Q

                magnitude = Q_prev2 ** 2 + Q_prev ** 2 - Q_prev * Q_prev2 * coeff
                results[freq].append(magnitude)

        return results

    def plot_raw_data(df, columns, output_file=None):
        fig = go.Figure()
        for col in columns:
            fig.add_trace(go.Scatter(x=df['Time'], y=df[col], mode='lines', name=col))
        fig.update_layout(
            title="Raw Data",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            legend_title="Columns",
        )
        if output_file:
            fig.write_html(output_file, auto_open=True)

    def plot_spectrogram(f, t, values, title, output_file=None):
        magnitude_db = 10 * np.log10(values + 1e-12)
        fig = go.Figure(
            data=go.Heatmap(
                z=magnitude_db,
                x=t,
                y=f,
                colorscale='Jet',
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            coloraxis_colorbar=dict(title="Amplitude (dB)"),
        )
        if output_file:
            fig.write_html(output_file, auto_open=True)

    def plot_goertzel_results(results, window_size, sampling_rate, output_file=None):
        time_step = window_size / sampling_rate
        time_bins = np.arange(len(next(iter(results.values())))) * time_step

        fig = go.Figure()
        for freq, magnitudes in results.items():
            fig.add_trace(go.Scatter(x=time_bins, y=magnitudes, mode='lines', name=f"{freq} Hz"))
            fig.add_trace(
                go.Scatter(x=time_bins, y=10 * np.log10(magnitudes), mode='lines', name=f"{freq} Hz (log scale)"))
        fig.update_layout(
            title="Goertzel Algorithm Results",
            xaxis_title="Time (s)",
            yaxis_title="Magnitude",
            legend_title="Frequencies",
        )
        if output_file:
            fig.write_html(output_file, auto_open=True)

    def calc(input_csv, output_folder, selected_columns, sampling_rate, f_resolution, t_resolution, overlap,
             downsample_factor, target_freqs, window_size):
        df = pd.read_csv(input_csv)

        df, sampling_rate = preprocess_data(df, sampling_rate, downsample_factor)

        if 'Time' not in df.columns:
            df['Time'] = np.arange(len(df)) / sampling_rate

        if "All" in selected_columns:
            selected_columns = [col for col in df.columns if col != 'Time']

        # Plot raw data
        plot_raw_data(df, selected_columns, f"{output_folder}/raw_data.html")

        for column in selected_columns:
            signal_column = df[column]

            # Compute STFT
            f_stft, t_stft, Zxx = compute_stft(signal_column, sampling_rate, f_resolution, t_resolution, overlap)
            plot_spectrogram(f_stft, t_stft, np.abs(Zxx), f"STFT Spectrogram - {column}",
                             f"{output_folder}/spectrogram_{column}.html")

            # Compute Goertzel Algorithm for selected frequencies
            goertzel_results = compute_goertzel(signal_column, sampling_rate, target_freqs, window_size)
            plot_goertzel_results(goertzel_results, window_size, sampling_rate,
                                  f"{output_folder}/goertzel_results_{column}.html")


    frame = ttk.Frame(tab2)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Frequency Analysis Tool", font=("Helvetica", 16)).pack(pady=10)

    # File selection
    def select_file():
        input_csv = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        if input_csv:
            file_label.config(text=input_csv)
            df = pd.read_csv(input_csv)
            column_dropdown['values'] = ["All"] + list(df.columns)
            column_dropdown.set("All")
            column_dropdown['state'] = 'readonly'

    file_label = tk.Label(frame, text="No file selected", fg="blue")
    file_label.pack()

    tk.Button(frame, text="Select File", command=select_file).pack()

    # Column dropdown
    tk.Label(frame, text="Select Column(s):").pack()
    column_dropdown = ttk.Combobox(frame, state="disabled")
    column_dropdown.pack()

    # Sampling Rate Dropdown
    tk.Label(frame, text="Sampling Rate (Hz):").pack()
    fs_dropdown = ttk.Combobox(frame, values=["50000/3", "12500", "1e6", "2e6", "Other"], state="readonly")
    fs_dropdown.pack()
    fs_dropdown.set("1e6")  # Default value

    def toggle_custom_fs(*args):
        if fs_dropdown.get() == "Other":
            entry_fs_custom.grid(row=2, column=1, padx=5, pady=5)
        else:
            entry_fs_custom.grid_remove()

    fs_dropdown.bind("<<ComboboxSelected>>", toggle_custom_fs)

    # Custom Fs Entry (Initially Hidden)
    entry_fs_custom = tk.Entry(frame)
    entry_fs_custom.pack()
    entry_fs_custom.pack_forget()

    # Frequency Resolution
    tk.Label(frame, text="Frequency Resolution (Hz):").pack()
    entry_f_resolution = tk.Entry(frame)
    entry_f_resolution.pack()
    entry_f_resolution.insert(0, "100")  # Default

    # Time Resolution
    tk.Label(frame, text="Time Resolution (s):").pack()
    entry_t_resolution = tk.Entry(frame)
    entry_t_resolution.pack()
    entry_t_resolution.insert(0, "0.01")  # Default

    # Downsample Factor
    tk.Label(frame, text="Downsample Factor:").pack()
    entry_downsample = tk.Entry(frame)
    entry_downsample.pack()
    entry_downsample.insert(0, "4")  # Default

    # Target Frequencies
    tk.Label(frame, text="Target Frequencies (comma-separated):").pack()
    entry_target_freqs = tk.Entry(frame)
    entry_target_freqs.pack()
    entry_target_freqs.insert(0, "6000")  # Default

    # Window Size
    tk.Label(frame, text="Window Size:").pack()
    entry_window_size = tk.Entry(frame)
    entry_window_size.pack()
    entry_window_size.insert(0, "64")  # Default

    # Overlap Checkbox
    var_overlap = tk.BooleanVar(value=False)
    tk.Checkbutton(frame, text="Use Overlap", variable=var_overlap).pack()

    # Add Progress Bar
    progress_bar = ttk.Progressbar(frame, mode="indeterminate", orient="horizontal", length=300)
    progress_bar.pack(pady=10)
    progress_bar.pack_forget()  # Initially hidden

    # Run Analysis
    def run_analysis():
        def analysis_task():
            try:
                # Get input CSV
                input_csv = file_label.cget("text")
                if not input_csv or input_csv == "No file selected":
                    raise ValueError("Please select a valid CSV file first!")

                # Get output folder
                output_folder = filedialog.askdirectory(title="Select Output Folder")
                if not output_folder:
                    raise ValueError("Please select an output folder!")

                # Get selected column
                selected_column = column_dropdown.get()
                if not selected_column:
                    raise ValueError("Please select a column!")

                # Get sampling rate
                fs_value = fs_dropdown.get()
                if fs_value == "Other":
                    fs_value = float(entry_fs_custom.get())
                else:
                    fs_value = float(fs_value)

                # Get other parameters
                f_resolution = float(entry_f_resolution.get())
                t_resolution = float(entry_t_resolution.get())
                overlap = var_overlap.get()
                downsample_factor = int(entry_downsample.get())
                target_freqs = list(map(float, entry_target_freqs.get().split(',')))
                window_size = int(entry_window_size.get())

                # Perform analysis
                calc(
                    input_csv=input_csv,
                    output_folder=output_folder,
                    selected_columns=[selected_column] if selected_column != "All" else ["All"],
                    sampling_rate=fs_value,
                    f_resolution=f_resolution,
                    t_resolution=t_resolution,
                    overlap=overlap,
                    downsample_factor=downsample_factor,
                    target_freqs=target_freqs,
                    window_size=window_size
                )

                # Show success message
                if messagebox.askyesno("Open Folder", "Analysis completed! Do you want to open the results folder?"):
                    webbrowser.open(output_folder)

            except Exception as e:
                messagebox.showerror("Error", str(e))

            finally:
                # Stop progress bar
                progress_bar.stop()
                progress_bar.pack_forget()

        # Start progress bar
        progress_bar.pack()
        progress_bar.start(10)

        # Run analysis in a separate thread
        threading.Thread(target=analysis_task).start()

    # Run button
    tk.Button(frame, text="Run Analysis", command=run_analysis).pack(pady=10)

# File Search and Plot Tool
def file_search_tool():
    selected_files = []
    selected_folder = ""
    Secondy_axis=False
    def select_folder_and_search():
        nonlocal selected_files, selected_folder,Secondy_axis
        folder_path = filedialog.askdirectory(title="Select Folder")
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder!")
            return
        Secondy_axis=Secondy_axis_bool.get()
        print(Secondy_axis)
        selected_folder = folder_path
        search_name = entry_search_name.get()
        if not search_name:
            messagebox.showerror("Error", "Please provide a name to search for!")
            return

        selected_files = []
        for file in os.listdir(folder_path):
            if search_name.lower() in file.lower():
                selected_files.append(file)

        if selected_files:
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, "\n".join(selected_files))
        else:
            messagebox.showinfo("No Files Found", f"No files found with the name '{search_name}'.")

    def plot_files():
        nonlocal selected_files, selected_folder,Secondy_axis
        if not selected_files or not selected_folder:
            messagebox.showerror("Error", "Please search and select files first!")
            return

        try:
            Secondy_axis = Secondy_axis_bool.get()
            fs_input = entry_fs.get()
            try:
                fs = eval(fs_input)
                if not isinstance(fs, (int, float)) or fs <= 0:
                    raise ValueError
            except Exception:
                raise ValueError("Invalid sampling rate")

            progress_text.delete("1.0", tk.END)
            for file_name in selected_files:
                plot_spi_files(selected_folder, [file_name], fs, progress_text,Secondy_axis)

            progress_text.insert(tk.END, "Plotting Completed.\n")
            progress_text.see(tk.END)
            progress_text.update()

            messagebox.showinfo("Plotting Completed", "Plots have been generated and saved.")

            if messagebox.askyesno("Open Folder", "Plots completed! Do you want to open the results folder?"):
                webbrowser.open(selected_folder)

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid sampling rate (fs)! Example: 50000/3 or 12500.")

    frame = ttk.Frame(tab1)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="File Search and Plot Tool", font=("Helvetica", 16)).pack(pady=10)

    search_frame = tk.Frame(frame)
    search_frame.pack(fill="x", padx=10, pady=5)

    tk.Label(search_frame, text="Search Name:").grid(row=0, column=0, sticky="w")
    entry_search_name = tk.Entry(search_frame, width=50)  # Increased width
    entry_search_name.grid(row=0, column=1, padx=5, pady=5)

    tk.Button(search_frame, text="Search", command=select_folder_and_search).grid(row=0, column=2, padx=5, pady=5)

    result_text = tk.Text(search_frame, height=10, width=80)  # Increased width
    result_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

    plot_frame = tk.Frame(frame)
    plot_frame.pack(fill="x", padx=10, pady=5)

    tk.Label(plot_frame, text="Sampling Rate (fs):").grid(row=0, column=0, sticky="w")
    entry_fs = tk.Entry(plot_frame, width=30)  # Increased width
    entry_fs.grid(row=0, column=1, padx=5, pady=5)
    entry_fs.insert(0, "50000/3")

    Secondy_axis_bool = tk.BooleanVar()
    tk.Checkbutton(plot_frame, text="Secondary Axis", variable=Secondy_axis_bool).grid(row=0, column=2, padx=5, pady=5)

    tk.Button(plot_frame, text="Plot", command=plot_files).grid(row=0, column=3, padx=5, pady=5)

    progress_text = tk.Text(plot_frame, height=10, width=80)  # Increased width
    progress_text.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

# Main Application
def open_gui():
    root = tk.Tk()
    root.title("Analysis Tools")

    notebook = ttk.Notebook(root)
    global tab1, tab2

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)

    notebook.add(tab1, text="File Search and Plot")
    notebook.add(tab2, text="Original Tool")

    notebook.pack(expand=1, fill="both")

    file_search_tool()
    original_tool()

    root.mainloop()

if __name__ == "__main__":
    open_gui()