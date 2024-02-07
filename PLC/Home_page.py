import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
import math
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import random
class GoertzelSampleBySample_spin:
    """Helper class to do Goertzel algorithm sample by sample"""
    def __init__(self, freq, fsamp, nsamp, alpha, window_bool):
        """I need Frequency, sampling frequency and the number of
        samples that we shall process"""
        self.freq, self.fsamp, self.nsamp = freq, fsamp, nsamp
        self.main_freq = freq
        self.consine_spin = 2 * math.pi * self.main_freq / fsamp
        self.koef = 2 * math.cos(self.consine_spin)
        self.cnt_samples = 0
        self.zprev1 = self.zprev2 = 0
        self.Erg = 0
        self.ErgFiltered = 0
        self.alpha = alpha
        self.Enable_prints = False
        self.enable_window = window_bool
        self.window = np.hamming(nsamp)
        self.t = 0
        self.slow_freq = self.main_freq / 100
        duration = 10  # Duration of the sine wave in seconds
        # Generate sample points
        self.time = np.arange(0, duration, 1 / fsamp)
        self.slow_sine = 5000 * np.sin(2 * np.pi * self.slow_freq * self.time)
        self.freq_arr2=[]

    def process_sample(self, samp):
        """Do one sample. Returns dBm of the input if this is the final
        sample, or None otherwise."""
        if self.enable_window:
            samp = samp * self.window[self.cnt_samples]
        Z = self.koef * self.zprev1 - self.zprev2 + (samp)
        self.zprev2 = self.zprev1
        self.zprev1 = Z
        self.cnt_samples += 1
        self.t += 1
        self.freq_arr2.append(self.freq)
        if self.cnt_samples == self.nsamp:
            self.Erg = self.zprev1 ** 2 + self.zprev2 ** 2 - self.koef * self.zprev1 * self.zprev2
            self.ErgFiltered = self.Erg  # =(self.alpha*self.Erg +(1-self.alpha)*self.ErgFiltered)/self.cnt_samples
            self.cnt_samples = 0
            return 1
        return None

    def update_freq(self):
        # self freq is the freq of the signal add a slow sine wavw to it
        self.freq = self.main_freq + self.slow_sine[self.t - 1]
        # print(self.t)
        #print(self.freq)

        self.consine_spin = 2 * math.pi * self.freq / self.fsamp
        self.koef = 2 * math.cos(self.consine_spin)

    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = 0
        self.zprev2 = 0
        self.update_freq()

class GoertzelSampleBySample:
    """Helper class to do Goertzel algorithm sample by sample"""
    def __init__(self, freq, fsamp, nsamp,alpha,window_bool):
        """I need Frequency, sampling frequency and the number of
        samples that we shall process"""
        self.freq, self.fsamp, self.nsamp = freq, fsamp, nsamp

        self.consine_spin=2 * math.pi * freq / fsamp
        self.koef = 2 * math.cos(self.consine_spin)
        self.cnt_samples = 0
        self.zprev1 = self.zprev2 = 0
        self.Erg=0
        self.ErgFiltered=0
        self.alpha = alpha
        self.Enable_prints = False
        self.enable_window=window_bool
        self.window = np.hamming(nsamp)
    def process_sample(self, samp):
        """Do one sample. Returns dBm of the input if this is the final
        sample, or None otherwise."""
        if self.enable_window:
            samp = samp * self.window[self.cnt_samples]
        Z=self.koef*self.zprev1 - self.zprev2+ (samp)
        self.zprev2 = self.zprev1
        self.zprev1=Z
        self.cnt_samples += 1
        if self.cnt_samples == self.nsamp:

            self.Erg=self.zprev1 ** 2 + self.zprev2 ** 2 - self.koef * self.zprev1 * self.zprev2
            self.ErgFiltered=(self.alpha*self.Erg +(1-self.alpha)*self.ErgFiltered)
            self.cnt_samples = 0
            return 1
        return None
    def reset(self):
        """Reset for a new calculation"""
        self.zprev1 = 0
        self.zprev2 = 0

def goertzel_freq_detect(sig,fs,detect_freq,N,alpha,Wind_bool):
    # Create Goertzel object
    goertzel = GoertzelSampleBySample(detect_freq, fs, N,alpha,Wind_bool)
    # Process samples
    dft_out_goertzel_1_TONE=[]
    #Time_axis_1_TONE=[]
    for idx,sample in enumerate(sig):
        temp = goertzel.process_sample(sample)
        if temp is not None:
            dft_out_goertzel_1_TONE.append(goertzel.ErgFiltered)
            #Time_axis_1_TONE.append(Time_array[idx])
            goertzel.reset()
    # Get dBm
    return 10*np.log10(dft_out_goertzel_1_TONE)


def goertzel_freq_detect_with_spin(sig,fs,detect_freq,N,alpha,Wind_bool):
    # Create Goertzel object
    goertzel = GoertzelSampleBySample_spin(detect_freq, fs, N,alpha,Wind_bool)
    # Process samples
    dft_out_goertzel_1_TONE=[]
    #Time_axis_1_TONE=[]
    for idx,sample in enumerate(sig):
        temp = goertzel.process_sample(sample)
        if temp is not None:
            dft_out_goertzel_1_TONE.append(goertzel.ErgFiltered)
            #Time_axis_1_TONE.append(Time_array[idx])
            goertzel.reset()

    # Get dBm
    return 10*np.log10(dft_out_goertzel_1_TONE)



def goertzel_freq_detect_with_skip(sig,fs,detect_freq,n,alpha,wind_bool,process_samples,skip_samples):
    # Create Goertzel Object
    goertzel = GoertzelSampleBySample(detect_freq, fs, n,alpha,wind_bool)
    # Process Samples
    dft_out_goertzel_1_tone=[]
    #time_axis_1_tone=[]
    samples_per_ms = fs // 1000
    total_samples = 0
    skip = False
    for idx, sample in enumerate(sig):
        if skip:
            total_samples += 1
            if total_samples % skip_samples == 0:
                skip = False
            continue
        temp = goertzel.process_sample(sample)
        if temp is not None:
            dft_out_goertzel_1_tone.append(goertzel.ErgFiltered)
            #time_axis_1_tone.append(time_array[idx])
            goertzel.reset()
        total_samples += 1
        if total_samples % process_samples == 0:
            skip = True
    # Get dBm
    return 10*np.log10(dft_out_goertzel_1_tone)




df_arc = pd.DataFrame({'Time': np.arange(0, 40, 0.1), 'RX': [0, 0, 2, 2, 0, 0, 4, 4] * 50})
df_arc['Time']=df_arc['Time']-df_arc['Time'][0]
# Done to change the data .... df_fa will be used in Page 2
Fs = 200000
Ds_fs = Fs
df_fa =  pd.read_parquet(r"E:\PycharmProjects\opt_roof_optimal_detection\combinedFa200K.parquet")
df_fa['Time']=df_fa['Time']-df_fa['Time'][0]
# Done to change the data .... df_fa will be used in Page 2

defualt_freq=40000
record_columns_arc = [col for col in df_arc.columns if 'rec' in col.lower()]

record_columns_FA = [col for col in df_fa.columns if 'rec' in col.lower()]


app = dash.Dash(__name__, suppress_callback_exceptions=True)
print('New sampling rate is: ', Ds_fs)
fs = Ds_fs
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
arc_layout = html.Div([
    html.H1('ARC Page'),
    html.Div(id='arc-content'),
    dcc.Link('Go to FA Page\n', href='/fa'),
    html.Label('Select Record:', style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='record-dropdown',
        options=[{'label': rec, 'value': rec} for rec in record_columns_arc],
        value=random.choice(record_columns_arc)   # Default value
    ),
    html.Label('Select Detection Frequency (m):', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='m-slider',
        min=1,
        max=int(Ds_fs / 2),  # Set the max value according to your use.
        value=defualt_freq,
    ),
    html.Label('Select N:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='n-slider',
        min=1,
        max=5000,  # Set the max value according to your use.
        value=128,
    ),
    html.Label('Select alpha:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='alpha-slider',
        min=0.1,
        max=1,  # Set the range according to your use.
        step=0.01,
        value=0.5,
    ),
    html.Label('Select avg:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='avg-slider',
        min=1,
        max=128,  # Set the range according to your use.
        step=1,
        value=16,
    ),
    html.Label('Wind bool:', style={'font-weight': 'bold'}),
    dcc.RadioItems(
        id='wind-bool',
        options=[{'label': str(i), 'value': str(i)} for i in [True, False]],
        value='True'
    ),
    dcc.Graph(id='Arc-graph'),
])

fa_layout = html.Div([
    html.H1('FA Page'),
    html.Div(id='fa-content'),
    dcc.Link('Go back to ARC Page', href='/arc'),
    html.Label('Select Record:', style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='record-dropdown',
        options=[{'label': rec, 'value': rec} for rec in record_columns_FA],
        value=random.choice(record_columns_FA)  # Default value
    ),
    html.Label('Select Detection Frequency (m):', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='m-slider',
        min=1,
        max=int(Ds_fs / 2),  # Set the max value according to your use.
        value=defualt_freq,
    ),
    html.Label('Select N:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='n-slider',
        min=1,
        max=5000,  # Set the max value according to your use.
        value=128,
    ),
    html.Label('Select alpha:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='alpha-slider',
        min=0.1,
        max=1,  # Set the range according to your use.
        step=0.01,
        value=0.5,
    ),
    html.Label('Select avg:', style={'font-weight': 'bold'}),
    dcc.Slider(
        id='avg-slider',
        min=1,
        max=128,  # Set the range according to your use.
        step=1,
        value=16,
    ),
    html.Label('Wind bool:', style={'font-weight': 'bold'}),
    dcc.RadioItems(
        id='wind-bool',
        options=[{'label': str(i), 'value': str(i)} for i in [True, False]],
        value='True'
    ),
    dcc.Graph(id='fa-graph'),
])


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)

def display_page(pathname):
    if pathname == '/fa':
        return fa_layout
    else:
        return arc_layout

@app.callback(
    Output('Arc-graph', 'figure'),
    [Input('record-dropdown', 'value'),
    Input('m-slider', 'value'),
    Input('n-slider', 'value'),
    Input('alpha-slider', 'value'),
    Input('avg-slider', 'value'),
    Input('wind-bool', 'value')],
    prevent_initial_call=False)

def update_figure(selected_record, selected_m, selected_n, selected_alpha, selected_avg, wind_bool):
    # print("-------------------------------------------")
    # print("selected_record: ", selected_record)
    # print("selected_Freq: ", selected_m)
    # print("selected_N: ", selected_n)
    # print("Selected alpha: ", selected_alpha)
    # # print into the dash
    # if pathname == '/arc':
        # Process df_arc here and display data
    process_samples = 680
    skip_samples = 100
    selected_data = df_arc[selected_record]
    # We convert the wind_bool back to boolean here
    wind_bool = wind_bool == 'True'

    dft_out_goertzel_1_tone = goertzel_freq_detect(selected_data, fs, selected_m, selected_n, selected_alpha, wind_bool)
    dft_out_goertzel_1_tone_spin = goertzel_freq_detect_with_spin(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool)
    dft_out_goertzel_1_tone_skip = goertzel_freq_detect_with_skip(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool, process_samples,
                                                                  skip_samples)
    # averages
    dft_out_goertzel_1_tone_avg = [np.mean(dft_out_goertzel_1_tone[i:i + selected_avg]) for i in
                                   range(0, len(dft_out_goertzel_1_tone) - selected_avg)]
    dft_out_goertzel_1_tone_avg_spin = [np.mean(dft_out_goertzel_1_tone_spin[i:i + selected_avg]) for i in
                                        range(0, len(dft_out_goertzel_1_tone_spin) - selected_avg)]
    dft_out_goertzel_1_tone_avg_skip = [np.mean(dft_out_goertzel_1_tone_skip[i:i + selected_avg]) for i in
                                        range(0, len(dft_out_goertzel_1_tone_skip) - selected_avg)]

    dft_out_goertzel_1_tone_spin = goertzel_freq_detect_with_spin(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool)

    trace1 = go.Scatter(y=dft_out_goertzel_1_tone, mode='lines', name='dft output')
    trace2 = go.Scatter(y=dft_out_goertzel_1_tone_spin, mode='lines', name='dft output spin')
    trace3 = go.Scatter(y=dft_out_goertzel_1_tone_skip, mode='lines', name='dft output skip')
    # create avg traces
    trace_avg1 = go.Scatter(y=dft_out_goertzel_1_tone_avg, mode='lines', name='dft output_avg')
    trace_avg2 = go.Scatter(y=dft_out_goertzel_1_tone_avg_spin, mode='lines', name='dft output_avg_spin')
    trace_avg3 = go.Scatter(y=dft_out_goertzel_1_tone_avg_skip, mode='lines', name='dft output_avg_skip')
    fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(trace1, row=1, col=1)
    # fig.add_trace(trace2, row=1, col=1)
    # fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace_avg1, row=1, col=1)
    fig.add_trace(trace_avg2, row=1, col=1)
    fig.add_trace(trace_avg3, row=1, col=1)

    title = "Arc Graph for record = {0}, m = {1}, n = {2}, alpha = {3}, average = {4}, window = {5}".format(
        selected_record, selected_m, selected_n, selected_alpha, selected_avg, wind_bool)
    fig.update_layout(title=title)
    fig.update_xaxes(title='Samples')  # replace with appropriate label
    fig.update_yaxes(title='dB')  # replace with appropriate label
    # fig.add_trace(trace3, row=1, col=1)
    return fig  # {'data': traces}


@app.callback(
    Output('fa-graph', 'figure'),
    [Input('record-dropdown', 'value'),
    Input('m-slider', 'value'),
    Input('n-slider', 'value'),
    Input('alpha-slider', 'value'),
    Input('avg-slider', 'value'),
    Input('wind-bool', 'value')],
    prevent_initial_call=True)
def update_fa_figure(selected_record, selected_m, selected_n, selected_alpha, selected_avg, wind_bool):
    process_samples = 680
    skip_samples = 100
    selected_data = df_fa[selected_record]
    # We convert the wind_bool back to boolean here
    wind_bool = wind_bool == 'True'

    dft_out_goertzel_1_tone = goertzel_freq_detect(selected_data, fs, selected_m, selected_n, selected_alpha, wind_bool)
    dft_out_goertzel_1_tone_spin = goertzel_freq_detect_with_spin(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool)
    dft_out_goertzel_1_tone_skip = goertzel_freq_detect_with_skip(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool, process_samples,
                                                                  skip_samples)
    # averages
    dft_out_goertzel_1_tone_avg = [np.mean(dft_out_goertzel_1_tone[i:i + selected_avg]) for i in
                                   range(0, len(dft_out_goertzel_1_tone) - selected_avg)]
    dft_out_goertzel_1_tone_avg_spin = [np.mean(dft_out_goertzel_1_tone_spin[i:i + selected_avg]) for i in
                                        range(0, len(dft_out_goertzel_1_tone_spin) - selected_avg)]
    dft_out_goertzel_1_tone_avg_skip = [np.mean(dft_out_goertzel_1_tone_skip[i:i + selected_avg]) for i in
                                        range(0, len(dft_out_goertzel_1_tone_skip) - selected_avg)]

    dft_out_goertzel_1_tone_spin = goertzel_freq_detect_with_spin(selected_data, fs, selected_m, selected_n,
                                                                  selected_alpha, wind_bool)

    trace1 = go.Scatter(y=dft_out_goertzel_1_tone, mode='lines', name='dft output')
    trace2 = go.Scatter(y=dft_out_goertzel_1_tone_spin, mode='lines', name='dft output spin')
    trace3 = go.Scatter(y=dft_out_goertzel_1_tone_skip, mode='lines', name='dft output skip')
    # create avg traces
    trace_avg1 = go.Scatter(y=dft_out_goertzel_1_tone_avg, mode='lines', name='dft output_avg')
    trace_avg2 = go.Scatter(y=dft_out_goertzel_1_tone_avg_spin, mode='lines', name='dft output_avg_spin')
    trace_avg3 = go.Scatter(y=dft_out_goertzel_1_tone_avg_skip, mode='lines', name='dft output_avg_skip')
    fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(trace1, row=1, col=1)
    # fig.add_trace(trace2, row=1, col=1)
    # fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace_avg1, row=1, col=1)
    fig.add_trace(trace_avg2, row=1, col=1)
    fig.add_trace(trace_avg3, row=1, col=1)

    title = "Fa Graph for record = {0}, m = {1}, n = {2}, alpha = {3}, average = {4}, window = {5}".format(
        selected_record, selected_m, selected_n, selected_alpha, selected_avg, wind_bool)
    fig.update_layout(title=title)
    fig.update_xaxes(title='Samples')  # replace with appropriate label
    fig.update_yaxes(title='dB')  # replace with appropriate label
    # fig.add_trace(trace3, row=1, col=1)
    return fig  # {'data': traces}

if __name__ == '__main__':
    app.run_server(host= '0.0.0.0',debug=False)