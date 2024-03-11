import os
import numpy as np
import pandas as pd
if os.getlogin() == "eddy.a":
    from my_pyplot import omit_plot as _O, plot as _P, print_chrome as _PC, clear as _PP, print_lines as _PL, send_mail as _SM
    import Plot_Graphs_with_Sliders as _G
    import my_tools


def read_raw_spi_binary_file(full_file_path, channels):
    """
    Takes a binary file, and returns a DataFrame.
    This may take a while... because it is fully flexible.
    Args:
        full_file_path: self-explanatory.
        channels: List of np.dtype; needs to be the same length as the shape expected.
    Returns:
        pd.DataFrame
    """
    with open(full_file_path, 'rb') as f:
        binary_data = f.read()

    if len(binary_data) % 2 != 0:
        print('len(data) % 2 != 0')
        binary_data = binary_data[1:-1]

    arranged_data = []
    for i in range(len(channels)):
        arranged_data.append(bytearray())
    channel_type_lengths = [np.array(0, dtype=dt).itemsize for dt in channels]

    index_data = 0
    index_channel = 0
    while index_data < len(binary_data):
        half_type_length = int(channel_type_lengths[index_channel] / 2)
        chunk = binary_data[index_data:index_data + half_type_length * 2]
        arranged_data[index_channel].extend(chunk[half_type_length:] + chunk[:half_type_length])

        index_data += half_type_length * 2
        index_channel += 1
        if index_channel == len(channels):
            index_channel = 0

    data = []
    for arr_data, dtype in zip(arranged_data, channels):
        data.append(np.frombuffer(arr_data, dtype))
    return pd.DataFrame(np.array(data).T, columns=[f'CH{ch + 1}' for ch in range(len(data))])


def read_raw_spi_binary_file_fast(full_file_path, channels):
    """
    Same as above, but all channels need to be the same length (16 bit, 32 bit, does not matter).
    Args:
        full_file_path: self-explanatory.
        channels: List of np.dtype; needs to be the same length as the shape expected.
    Returns:
        pd.DataFrame
    """
    with open(full_file_path, 'rb') as f:
        binary_data = f.read()

    if len(binary_data) % 2 != 0:
        print('len(data) % 2 != 0')
        binary_data = binary_data[1:-1]

    arranged_data = []
    for i in range(len(channels)):
        arranged_data.append(bytearray())

    type_length = np.array(0, dtype=channels[0]).itemsize
    half_type_length = int(type_length / 2)
    index_channel = 0
    for i in range(0, len(binary_data), type_length):
        chunk = binary_data[i:i + type_length]
        arranged_data[index_channel].extend(chunk[half_type_length:] + chunk[:half_type_length])
        index_channel += 1
        if index_channel == len(channels):
            index_channel = 0

    data = []
    for arr_data, dtype in zip(arranged_data, channels):
        data.append(np.frombuffer(arr_data, dtype))
    return pd.DataFrame(np.array(data).T, columns=[f'CH{ch + 1}' for ch in range(len(data))])


def read_raw_spi_binary_file_faster(full_file_path, data_type, num_channels):
    """
    Same as above, but all channels need to be the same dtype.
    Args:
        full_file_path: self-explanatory.
        data_type: List of np.dtype; needs to be the same length as the shape expected.
        num_channels: number of channels.
    Returns:
        pd.DataFrame
    """
    with open(full_file_path, 'rb') as f:
        binary_data = f.read()

    if len(binary_data) % 2 != 0:
        print('len(data) % 2 != 0')
        binary_data = binary_data[1:-1]

    type_length = np.array(0, dtype=data_type).itemsize
    half_type_length = int(type_length / 2)
    data = bytearray()
    for i in range(0, len(binary_data), type_length):
        chunk = binary_data[i:i + type_length]
        data.extend(chunk[half_type_length:] + chunk[:half_type_length])

    data = np.frombuffer(data, data_type)
    data = data.reshape(-1, num_channels)
    return pd.DataFrame(data, columns=[f'CH{ch + 1}' for ch in range(num_channels)])


file_path = r"bin_6_F.bin"
df1 = read_raw_spi_binary_file(file_path, channels=[np.float32] * 6)
file_path = r"4f_2int.bin"
df2 = read_raw_spi_binary_file(file_path, channels=[np.float32] * 4 + [np.int32] * 2)
file_path = r"bin_6_F.bin"
df3 = read_raw_spi_binary_file_fast(file_path, channels=[np.float32] * 6)
file_path = r"bin_6_F.bin"
df4 = read_raw_spi_binary_file_faster(file_path, data_type=np.float32, num_channels=6)
