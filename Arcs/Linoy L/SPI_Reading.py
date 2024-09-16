import dataclasses
import datetime
import os
import re
import sys
import time
import matplotlib
import numpy
import pandas as pd
from matplotlib import pyplot as plt
import math


def read_file(file_path):
    """
    reads the values from a spi log_file
    :param file_path: path to the spi log file
    :return: values => values for them
    """
    # initialize these to empty lists
    Iac_L1_VEC = []
    Iac_L2_VEC = []
    Iac_L3_VEC = []
    RX_out_VEC = []
    Vdc_fast_VEC = []
    Vac_L1_VEC = []
    with open(file_path, "r") as filer:
        for i_line, line in enumerate(filer):
            if (len(line) > 10) and ("," in line):
                pcs = line.strip().split(",")
                try:
                    Iac_L1 = float(pcs[0])
                    Iac_L2 = float(pcs[1])
                    Iac_L3 = float(pcs[2])
                    RX_out = float(pcs[3])
                    Vdc_fast = float(pcs[4])
                    Vac_L1 = float(pcs[5])
                    Iac_L1_VEC.append(Iac_L1)
                    Iac_L2_VEC.append(Iac_L2)
                    Iac_L3_VEC.append(Iac_L3)
                    RX_out_VEC.append(RX_out)
                    Vdc_fast_VEC.append(Vdc_fast)
                    Vac_L1_VEC.append(Vac_L1)
                except:
                    if i_line != 0:
                        print(f'file_path = {file_path}, index line = {i_line}: line = {line.strip()}')

    return Iac_L1_VEC, Iac_L2_VEC, Iac_L3_VEC, RX_out_VEC, Vdc_fast_VEC, Vac_L1_VEC
