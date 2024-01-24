import numpy as np
import cmath
import tkinter as tk
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import os
from tkinter import filedialog
import shutil
import math ,sys
from tkinter import ttk
import time
import datetime
import statistics
from numpy.linalg import inv
import pylab
from numpy import arange
from numpy import sin
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
import plotly.graph_objs as go

############### LS Function ###############

def LS(H,y):
    H = np.mat(H)
    HtH = np.matmul(H.T, H)
    inv_HtH = inv(HtH)
    inv_HtH_Ht = np.matmul(inv_HtH, H.T)
    inv_HtH_Ht_y = np.matmul(inv_HtH_Ht, y)
    para = inv_HtH_Ht_y
    return(inv_HtH_Ht_y)

def Ls_phase_est(N=100,w=0.064,file_name=''):

    row = 3
    col = 1
    plots_per_pane = 3
    fig = initialize_fig(row=row, col=col, plots_per_pane=plots_per_pane, shared_xaxes=True,
                         subplot_titles=['Signal','Phase', 'Amplitude'])
    input=np.array([])
    for file in files:
        print(file)
        df = pd.read_csv(file).dropna(how='all', axis='columns')
        for index_df, (title_df, sub_df) in enumerate(df.items()):
            if title_df == "ADC in (RX1)":
                input = sub_df.to_numpy().flatten()

        input_time=np.arange(0,len(input.flatten()))/50e3
        y = input.flatten()
        y_sliced = y[0:(len(y) // N) * N]
        y = np.reshape(y_sliced, ((int(len(y_sliced) / N)),N))
        y = y.T

        # init matrix H
        H = np.zeros((N, 2))
        Jump = 0

        para_x,para_y=[],[]
        for z in range(0, len(y_sliced) // N):
            for k in range(0, N):
                H[k] = [np.sin(w * ((k) + Jump)), np.cos(w * ((k) + Jump))]  # building the row of the matrix.
            Jump += N
            Vec1=y.T[z]
            para=LS(H, Vec1)
            theta = np.arctan(para[0,1]/para[0,0])
            para_y.append(para[0,1])
            para_x.append(para[0,0])




        para_x = np.array(para_x)
        para_y = np.array(para_y)
        para_x_g = para_x[1:1000]
        para_y_g = para_y[1:1000]

        comlex_from_para_g=(para_x_g+1j*para_y_g).T;
        ### Second LS
        H2 = np.array([comlex_from_para_g[1:]]).T
        Vec2 = np.array(comlex_from_para_g[:-1].T)
        Filter_exp_val=LS(H2,Vec2)
        complex_from_para = (para_x + 1j * para_y);

        #filter = [1, -Filter_exp_val.H]
        #filter_applied = np.convolve(filter, comlex_from_para)

        exp_fix = (Filter_exp_val.getA() ** np.array(range(len(complex_from_para))))[0] #illuminate delta omega
        phase_jump = exp_fix* complex_from_para;
        phase_jump_no_transient_effect = phase_jump[10:]/ phase_jump[:-10]
        phase_unwarp = np.arctan(np.imag(phase_jump_no_transient_effect)/ np.real(phase_jump_no_transient_effect));
        Amp=abs(phase_jump_no_transient_effect)

        phase_time = np.arange(0, len(phase_unwarp)) / (50e3/N)
        fig.add_trace(go.Scattergl(x=input_time,y=input[:],
                                   name='input ', mode="lines",
                                   visible=False,
                                   line=dict(color="red"), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scattergl(x=phase_time,y=phase_unwarp*180/np.pi,
                                   name='Phase', mode="lines",
                                   visible=False,
                                   line=dict(color="blue"), showlegend=True), row=2, col=1)

        fig.add_trace(go.Scattergl(x=phase_time,y=Amp,
                                   name='Amp ', mode="lines",
                                   visible=False,
                                   line=dict(color="orange"), showlegend=True), row=3, col=1)











    for i in range(plots_per_pane):
        fig.data[i].visible = True
    steps = []
    col_names_list = files
    for i in range(0, int(len(fig.data) / plots_per_pane)):
        Temp = col_names_list[i]
        Temp = Temp[:Temp.find('Rec') + 6]
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
    fig.write_html('temp.html',
        config={'scrollZoom': True, 'responsive': False}, auto_open=True)



def initialize_fig(row = 4,col = 1,plots_per_pane=4,shared_xaxes=True,subplot_titles=['Rx','Rx','Rx','Rx']):

    all_specs = np.array([[{"secondary_y": True}] for x in range((row*col))])
    all_specs_reshaped = (np.reshape(all_specs, (col, row)).T).tolist()
    fig= make_subplots(rows=row, cols=col,specs=all_specs_reshaped, shared_xaxes=shared_xaxes,   subplot_titles=subplot_titles)

    return fig
### First, we will generate a test dataset.

def curve_fiting(x,y):

    # curve fit
    popt, _ = curve_fit(objective12, x, y)
    # summarize the parameter values
    a, b, c, d, e, f, g, h, i, k, r, n, m= popt
    print(popt)
    # plot input vs output
    pyplot.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = arange(min(x), max(x), 1)
    # calculate the output for the range
    y_line = objective12(x_line , a, b, c, d, e, f,g,h,i,k,r,n,m)
    # create a line plot for the mapping function
    pyplot.plot(x_line, y_line, '--', color='red')
    pyplot.show()

####### MAIN #######

N = 100
fs =50e3

w = 2*np.pi*(511)/fs
#w = 2*np.pi*(500)/fs

# path_log_folder = r'D:\projects\SPI_PLL'
# path_log_name = 'SPI Rec004.txt'
#
# file = f'{path_log_folder}/{path_log_name}'
#file=r"C:\Users\noam.d\PycharmProjects\LS-phase-est\Nimrodm.csv"
files=['Rec002.txt', 'Rec004.txt']
# Calling the LS function
#file=r'SPI Rec005.txt'

Ls_phase_est(N,w,files)
# df = pd.read_csv(file).dropna(how='all', axis='columns')
# df1=df.rolling(20).mean()
# df1.dropna(how='all', axis='columns',inplace=True)
# x=df1['freq']
# y=df1['Amp[V]']
# x = x[~np.isnan(x)]
# y = y[~np.isnan(y)]


print("\nDone!\n")
