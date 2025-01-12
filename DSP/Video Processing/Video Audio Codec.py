"""
Search all video files and check their audio codec.
"""

import os
import sys
import subprocess
from tabulate import tabulate


def get_audio_codec(file_path):
    # Run ffprobe to get file info
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1", file_path]
    try:
        # Run the subprocess and get the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            codec = result.stdout.strip().split("=")[-1]
            return codec
        else:
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def change_audio_codec(input_file, output_file):
    # If the audio codec is MPEG, convert it to AAC
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "copy",     # Copy the video stream without re-encoding
        "-c:a", "aac",      # Change audio codec to AAC
        "-b:a", "192k",     # Set audio bit-rate (optional)
        "-y",               # Overwrite output file without prompt
        output_file
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Audio codec changed successfully. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")


def find_mp4_files_and_audio_codecs(directory, change_mpeg_to_aac=False):
    table_data = []
    # Traverse directory and look for .mp4 files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                file_path = os.path.join(root, file)
                codec = get_audio_codec(file_path)

                if codec:
                    table_data.append([root, file, codec])
                    if change_mpeg_to_aac and (codec.lower() == "mpeg" or codec.lower() == "mp3"):
                        output_file = os.path.join(root, f"{file[:-4]} (aac converted).mp4")
                        change_audio_codec(file_path, output_file)
                        table_data.append([root, output_file, f"aac (converted)"])
                else:
                    table_data.append([root, file, "Unknown"])

    if table_data:
        if save_to_file[0]:
            print("\nPath\tFile Name\tCodec")
            [print(f"{"\t".join(row)}") for row in table_data]
        else:
            print("\n" + tabulate(table_data, headers=["Path", "File Name", "Codec"], tablefmt="csv" if save_to_file[0] else "grid"))
    else:
        print("No MP4 files found.")


if __name__ == "__main__":
    directory = r""
    change_mpeg_to_aac = False
    save_to_file = [False, rf"{directory}\Video Audio Codec - log.txt"]
    shutdown_pc = [False, 10]

    if save_to_file[0]:
        sys.stdout = open(save_to_file[1], "w", encoding='utf-8')
    find_mp4_files_and_audio_codecs(directory, change_mpeg_to_aac)
    if save_to_file[0]:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    if shutdown_pc[0]:
        os.system(f"shutdown /s /f /t {shutdown_pc[1]}")    # /s = shutdown;   /f forces apps to close;   /t sets timer in seconds
