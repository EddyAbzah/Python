import numpy as np
import pandas as pd
import os
import subprocess as sps
import sys
import ltspice_ido
import time
import math
import datetime
import openpyxl as px
import tkinter as tk
from tkinter import filedialog
from si_prefix import si_format
import plotly.figure_factory as ff
import plotly.io as pio
import plotly as py
import threading
import shutil
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignores DeprecationWarning of LTspice addon

path_folder_temp = r'C:\Users\eddy.a\Downloads\TX1RX1\Mixer with LT Spice'
path_schematic = 'D1288_RX_Channel_TEST'  # LTSpice file to run (without .asc)
path_params = 'RX_Params-JupiterPlus'  # for Mixer manipulation
path_input_files = 'Arc_TX1RX1.csv'
run_time = 10
mixer_arr = [10, 20, 25, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]

for i_mixer, mixer in enumerate(mixer_arr):
    path_folder = f'{path_folder_temp} ({i_mixer})' + '\\'
    file_name = path_folder + path_schematic
    start_time = time.time()  # Start of simulation time counting
    if run_time > 0 or path_input_files != '':
        lines_to_write = []
        with open(file_name + ".asc", "r+") as f:
            for line in f:
                if run_time > 0 and 'TEXT' in line and '!.tran' in line:
                    lines_to_write.append(f"{line.split('tran 0 ')[0]}tran 0 {run_time} {line.split('tran 0 ')[1].split(' ')[1]}")
                elif path_input_files != '' and 'SYMATTR Value PWL' in line:
                    lines_to_write.append(f"SYMATTR Value PWL file={path_input_files}\n")
                else:
                    lines_to_write.append(line)
            f.seek(0)  # set the pointer to the beginning
            f.truncate()  # cutting off the rest
            f.write("".join(lines_to_write))
            f.close()
        print(f'LTspice run time changed to = {run_time}')
    if mixer > 0:
        lines_to_write = []
        with open(path_folder + path_params + ".txt", "r+") as f:
            for line in f:
                if '.param mix_freq  = ' in line:
                    lines_to_write.append(f"{line.split(' = ')[0]} = {si_format(mixer, format_str=u'{value}{prefix}')};\n")
                else:
                    lines_to_write.append(line)
            f.seek(0)  # set the pointer to the beginning
            f.truncate()  # cutting off the rest
            f.write("".join(lines_to_write))
            f.close()
        print(f'LTspice mixer changed to = {mixer}')

    print("\nStarting Simulation...")
    try:
        sps.run(["C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe", "-b", "-run", path_schematic + '.asc'], shell=True,
                check=True, stdout=sys.stdout, stderr=sps.STDOUT)  # Run simulation via command line in the same process
    except:
        print("\nThere was an ERROR running simulation {}. Exiting...\n")
        exit(-1)

    print("\n--> The process took: %s (Hours,Min,Sec)\n" % (
        datetime.timedelta(seconds=math.floor(time.time() - start_time))))  # Print the simulation elapsed time
    output = ltspice_ido.Ltspice(path_schematic + '.raw')  # Analyze the .RAW data
    print("\nSimulation ended. Analyzing Data...")
    output.parse()  # Analyze the data

    dict = {'Time': output.getTime(), 'RX_out': output.getData('V(out)')}  # Save csv file
    # Remove simulation files
    os.remove(file_name + '.raw')
    os.remove(file_name + '.net')
    os.remove(file_name + '.log')
    os.remove(file_name + '.op.raw')

    print("Saving Data to .CSV file\n")
    df = pd.DataFrame(dict)
    df.to_csv(f'LTspice OUT (TX1RX1 Mixer = {mixer}).csv')
    print("Done!\n")

    print("\n--> The process took: %s (Hours,Min,Sec)\n" % (
        datetime.timedelta(seconds=math.floor(time.time() - start_time))))  # Print the simulation elapsed time
