import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

def plot_with_slider(df,lab_folder,config,auto_open,kill_chrome_bool,channels,row, col,plots_per_pane,filename_out):
    fig = make_fig(row=1, col=1, subplot_titles=channels, shared_xaxes=True)
    Rec_arr = []
    max_res = 5000
    plot_spaces = int(len(df) / max_res)
    if plot_spaces == 0:
        plot_spaces = 1
    for col in df.columns:
        if col in channels:
            fig.add_trace(go.Scattergl(x=df[col][::plot_spaces],
                                       y=df[col][::plot_spaces],
                                       name=col + " " ,
                                       visible=False,
                                       showlegend=True), row=1, col=1)

    resutls_path = lab_folder + '\\' + 'kaka' + '\\'
    if not os.path.exists(resutls_path):
        os.makedirs(resutls_path)
    fig.update_layout(
        title=os.path.basename(lab_folder),
        font_family="Courier New",
        font_color="blue",
        font_size=20,
        title_font_family="Times New Roman",
        title_font_color="red",
        legend_title_font_color="green"
    )
    for i in range(plots_per_pane):
        fig.data[i].visible = True
    steps = []
    for i in range(0, int(len(fig.data) / plots_per_pane)):
        Temp = channels[i]

        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider  switched to step "}],
            label=str(Temp)  # layout attribute
        )
        j = i * plots_per_pane
        for k in range(plots_per_pane):
            step["args"][0]["visible"][j + k] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "REC: "},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders)

    py.offline.plot(fig, config=config, auto_open=auto_open,
                    filename=resutls_path + os.path.basename(lab_folder) + filename_out + '.html')
    kill_chrome(kill_chrome_bool)
    print('Done Zes')
def make_fig(row,col,subplot_titles,shared_xaxes):
    all_specs = np.array([[{"secondary_y": True}] for x in range(row * col)])

    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()

    fig = make_subplots(rows=row, cols=col,
                        specs=all_specs_reshaped, shared_xaxes=True, subplot_titles=subplot_titles)
    return fig


df=pd.read_csv(r"C:\Users\noam.d\Downloads\iris_csv.csv")
channels=["sepallength","sepalwidth","petalwidth"]
row=1
col=1
plots_per_pane=1
kill_chrome_bool=False
filename_out=' ido '
config = {'scrollZoom': True, 'responsive': False, 'editable': True, 'modeBarButtonsToAdd': ['drawline',
                                                                                             'drawopenpath',
                                                                                             'drawclosedpath',
                                                                                             'drawcircle',
                                                                                             'drawrect',
                                                                                             'eraseshape'
                                                                                             ]}
lab_folder=r'E:\Test ido'
auto_open=True


plot_with_slider(df,lab_folder,config,auto_open,kill_chrome_bool,channels,row, col,plots_per_pane,filename_out)
