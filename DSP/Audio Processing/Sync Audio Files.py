import os
import re
import librosa
import numpy as np
from tabulate import tabulate
from pydub import AudioSegment
from scipy.signal import correlate


def load_audio(file_path, trim_seconds_from_start, trim_to_length, target_sample_rate=48000):
    audio = AudioSegment.from_file(file_path)
    duration_before_trim = len(audio) / 1000

    if duration_before_trim > trim_seconds_from_start + trim_to_length:
        trim_start_ms = trim_seconds_from_start * 1000
        trim_length_ms = trim_to_length * 1000
        audio = audio[trim_start_ms:trim_start_ms + trim_length_ms]
    else:
        raise ValueError(f"Audio file '{file_path}' is too short to trim with "
                         f"{trim_seconds_from_start = } and/or {trim_to_length = }.")

    duration_after_trim = len(audio) / 1000
    print(f'Loaded: {os.path.basename(file_path)} (duration before/after trim = {duration_before_trim:.2f} / {duration_after_trim:.2f} seconds)')

    audio = audio.set_channels(1).set_frame_rate(target_sample_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples, target_sample_rate


def calculate_offset(signal1, signal2, sample_rate, hop_length=512):
    onset_env1 = librosa.onset.onset_strength(y=signal1, sr=sample_rate, hop_length=hop_length)
    onset_env2 = librosa.onset.onset_strength(y=signal2, sr=sample_rate, hop_length=hop_length)

    onset_env1 = (onset_env1 - np.mean(onset_env1)) / np.std(onset_env1)
    onset_env2 = (onset_env2 - np.mean(onset_env2)) / np.std(onset_env2)

    min_len = min(len(onset_env1), len(onset_env2))
    onset_env1 = onset_env1[:min_len]
    onset_env2 = onset_env2[:min_len]

    correlation = correlate(onset_env1, onset_env2, mode='full')
    lag = np.argmax(correlation) - (len(onset_env1) - 1)

    offset_seconds = lag * hop_length / sample_rate
    return offset_seconds


def print_results_table(results):
    headers = ["#", "File Name", "Offset (sec)", "Relation to Reference"]
    results = sorted(results, key=lambda x: x[0])
    table_data = []

    for (idx, filepath, offset) in results:
        filename = os.path.basename(filepath)
        if np.isnan(offset):
            relation = "Error loading"
            offset_str = "---"
        elif idx == reference_file_index:
            relation = "Is Reference"
            offset_str = "0.0"
        else:
            relation = (
                "Ahead of reference" if offset < 0 else
                "Behind reference" if offset > 0 else
                "Perfectly aligned"
            )
            offset_str = f"{offset:.3f}"
        table_data.append([idx, filename, offset_str, relation])

    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    # print("\n".join([f"{offset[2]:.3f}" for offset in results]))


if __name__ == "__main__":
    folder = r""
    filter_files = re.compile(r'.*\.mp3$', re.IGNORECASE)
    audio_files = [file for file in os.listdir(folder) if filter_files.match(file)]        # and "guitar pro" not in file.lower()
    audio_files.sort()

    reference_file_index = 0
    trim_seconds_from_start = 0
    trim_to_length = 0

    results = []

    try:
        reference = audio_files.pop(reference_file_index)
        ref_signal, ref_sr = load_audio(reference, trim_seconds_from_start, trim_to_length)
        results.append((reference_file_index, reference, 0))

        for file_index, path in enumerate(audio_files):
            if file_index >= reference_file_index:
                file_index += 1
            try:
                signal, sr = load_audio(path, trim_seconds_from_start, trim_to_length)
                if sr != ref_sr:
                    raise ValueError(f"Sample rate mismatch in {path}")

                offset = calculate_offset(ref_signal, signal, ref_sr)
                results.append((file_index, path, offset))

            except Exception as e:
                print(f"[!] Skipping {path}: {e}")
                results.append((path, float('nan')))

        print_results_table(results)

    except Exception as e:
        print(f"Critical Error: {e}")
