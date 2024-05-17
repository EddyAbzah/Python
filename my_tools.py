from multipledispatch import dispatch
import filecmp
import struct
import shutil
import re
import os


def delete_thumbs(folder):
    """
        Delete tricky files
    """
    # os.chdir(folder)
    os.rename(folder + 'Thumbs.db', folder + 'Thumbs.dbdb')
    os.remove(folder + 'Thumbs.dbdb')


def delete_folder(folder):
    """
        Delete folder
    """
    try:
        shutil.rmtree(folder)
    except:
        delete_thumbs(folder)
    finally:
        os.rmdir(folder)


@dispatch(str, str, str)
def compare_folders(main_folder, folder_1, folder_2):
    """
    Compares two folders file by file, and prints the results.
    Args:
        main_folder: Path of the two folders.
        folder_1: NAME of folder number 1.
        folder_2: NAME of folder number 2.
    Returns:
        None.
        Results are printed.
    """
    folder_1 = main_folder + '\\' + folder_1
    folder_2 = main_folder + '\\' + folder_2
    compare_folders(folder_1, folder_2)


@dispatch(str, str)
def compare_folders(folder_1, folder_2):
    """
    Compares two folders file by file, and prints the results.
    Args:
        folder_1: FULL path of folder number 1.
        folder_2: FULL path of folder number 2.
    Returns:
        None.
        Results are printed.
    """
    filecmp.dircmp(folder_1, folder_2).report_full_closure()


def find_numbers_in_text(text):
    """
    FInd numbers in strings using Regex.
    Args:
        text: random text.
    Returns:
        numbers as list of floats.
    """
    numbers = re.findall(r'-?\d+\.\d+', text)
    print(f'text = {text}')
    print(f'numbers = {[float(n) for n in numbers]}')
    return [float(n) for n in numbers]


def convert_df_counters(df):
    """
        Find the word 'counter' in df columns and bitwise convert them into ints.
    """
    counter_cols = [col for col in df.columns if 'counter' in col.lower()]
    df[counter_cols] = df[counter_cols].applymap(bitwise_float_to_int)
    return df


def bitwise_float_to_int(val):
    """
        Bitwise convert floats into ints. Returns the same value if ValueError.
    """
    try:
        return struct.unpack('>l', struct.pack('>f', val))[0]
    except ValueError:
        return val


def bitwise_int_to_float(val):
    """
        Bitwise convert ints into floats. Returns the same value if ValueError.
    """
    try:
        return struct.unpack('>f', struct.pack('>l', int(val)))[0]
    except ValueError:
        return val


def bitwise_ushort_to_short(val):
    """
        Bitwise convert uInt16 into Int16. Returns the same value if ValueError.
    """
    try:
        return struct.unpack('>h', struct.pack('>H', int(val)))[0]
    except ValueError:
        return val
