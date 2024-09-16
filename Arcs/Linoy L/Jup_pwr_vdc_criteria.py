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


def read_log_file(file_path):
    """
    reads the values from a log_file
    :param file_path: path to the log file
    :return: indices, values => indices and values for them
    """
    energies = []  # initialize these to empty lists
    IacVec = []
    VdcVec = []
    EnergyCounterVec = 0
    IacCounterVec = 0
    with open(file_path, "r") as filer:
        for i_line, line in enumerate(filer):
            if (len(line) > 45) and ("@@" in line):
              pcs = line.strip().split(",")
              try:
                Energy = float(pcs[2])
                Vdc = float(pcs[4])
                Iac = float(pcs[3])
                energies.append(Energy)
                IacVec.append(Iac)
                VdcVec.append(Vdc)
              except:
                  print(f'file_path = {file_path}, index line = {i_line}: line = {line}')
    energies = [e * 2 for e in energies]
    energies = 10 * numpy.log10(energies)
    return energies , EnergyCounterVec ,IacCounterVec, VdcVec,IacVec