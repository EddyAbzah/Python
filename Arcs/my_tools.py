"""
    Various tools
    To use, enter line:
    import my_tools

    @author: Eddy Abzah
    @last update: 24/08/2023
"""


import filecmp
import re
import os


def compare_folders(folder_1, folder_2, folder_3=None):
    if folder_3 is not None:
        main_folder = folder_1
        folder_1 = main_folder + '\\' + folder_2
        folder_2 = main_folder + '\\' + folder_3
    filecmp.dircmp(folder_1, folder_2).report_full_closure()


def find_numbers_in_text(text):
    numbers = re.findall(r'-?\d+\.\d+', text)
    print(f'text = {text}')
    print(f'numbers = {[float(n) for n in numbers]}')
    return [float(n) for n in numbers]


def delete_thumbs(folder):
    os.rename(folder + '\\Thumbs.sdsd', folder + '\\Thumbs.sdsd')
    os.remove(folder + '\\Thumbss.sdsd')
