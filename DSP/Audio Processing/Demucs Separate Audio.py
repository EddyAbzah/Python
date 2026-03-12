"""
Batch audio stem separation and processing using Demucs, with two possible models:
1. htdemucs: newer and faster
2. mdx_extra_q: older and slower, but perhaps better for guitar separation


This script:
1. Separates audio tracks into stems using Demucs
2. Mixes custom outputs (old files will be overwritten):
   - No Drums
   - Drums Only
   - Vocals Only
3. Optionally deletes intermediate stems
4. Applies MP3 metadata tags


Demucs repository:
https://github.com/adefossez/demucs


Python packages:
pip install demucs, pydub, tqdm, mutagen
"""


import re
import shutil
import traceback
import subprocess
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from datetime import datetime
from mutagen.easyid3 import EasyID3


input_folders = [
    Path(r"C:\Users\_____\Music\*"),
]
files_filter_in = "*.mp3"
files_filter_out = ["(ND", "(SD", "(SV", "Separated", "Mixes"]

output_folder_separated = Path(r"C:\Users\_____\Downloads\Demucs Separate Audio\Separated")
output_folder_mixed = Path(r"C:\Users\_____\Downloads\Demucs Separate Audio\Mixes")

enable_demucs = True
demucs_models = ["htdemucs", "mdx_extra_q"][:1]     # = ["htdemucs"]
output_types = ["wav", "mp3"][1:]                   # = ["mp3"]
mp3_bitrate = 128

enable_mix = True
no_drums_output = [True, "No Drums", "ND"]
drums_only_output = [True, "Solo Drums", "SD"]
vocals_only_output = [True, "Solo Vocals", "SV"]
no_drums_mix_levels = {"vocals": 0, "bass": 0, "other": 0}

enable_file_tagging = True
artist_name = "Various Practice"
year = 2026
track_counter = 1                                   # will increment per song

enable_unmixed_file_deletion  = False
enable_skip_if_file_exists = True
title_match_pattern = r"(\(\d{4}\)\s\d{2}\s)(.+)"
save_log = True
log_file_output = []


def log(line):
    """Log messages safely alongside tqdm bars."""
    tqdm.write(line)
    if save_log:
        log_file_output.append(line)


def get_source_tags(file: Path):
    """Extract artist and album from the original file."""
    try:
        audio = EasyID3(file)
        title = audio.get("title", ["Unknown Title"])[0]
        artist = audio.get("artist", ["Unknown Artist"])[0]
        album = audio.get("album", ["Unknown Album"])[0]
    except Exception:
        title = "Unknown Title"
        artist = "Unknown Artist"
        album = "Unknown Album"
    match = re.match(title_match_pattern, str(file.stem))
    if match:
        title = f"{artist} - {match.group(2)}"
    return title, artist, album


def run_demucs(file: Path):
    """Run Demucs separation for all models and output types."""
    for model in demucs_models:
        stem_folder = output_folder_separated / model / file.stem
        if enable_skip_if_file_exists and stem_folder.exists() and any(stem_folder.iterdir()):
            log(f"Skipping Demucs: {stem_folder}")
            continue
        for ext in output_types:
            cmd = ["demucs", "--device", "cuda", "-n", model, "--shifts", "1", "-j", "2"]
            if ext == "mp3":
                cmd += ["--mp3", "--mp3-bitrate", str(mp3_bitrate)]
            cmd += ["-o", str(output_folder_separated), str(file)]
            log("Running Demucs: " + " ".join(cmd))
            subprocess.run(cmd)


def mix_stems(file: Path, title: str, artist: str):
    """Mix stems into No Drums, Drums Only, Vocals Only, then export."""
    for model in demucs_models:
        for ext in output_types:
            song_name = file.stem
            stem_folder = output_folder_separated / model / song_name
            if not stem_folder.exists():
                log(f"Stem folder not found: {stem_folder}")
                continue

            # Load stems
            stems = {}
            for stem_file in stem_folder.glob(f"*.{ext}"):
                stem_name = stem_file.stem.lower()
                stems[stem_name] = AudioSegment.from_file(stem_file)

            # Prepare outputs
            outputs = [
                ("No Drums", no_drums_output, lambda: [k for k in stems if k != "drums"]),
                ("Drums Only", drums_only_output, lambda: ["drums"] if "drums" in stems else []),
                ("Vocals Only", vocals_only_output, lambda: ["vocals"] if "vocals" in stems else [])
            ]

            # Mix each output
            for name, (enabled, suffix_long, suffix_short), stem_keys_func in tqdm(outputs, desc=f"Mixing {song_name} ({model})", leave=False,
                                                                                   colour="green", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
                if not enabled:
                    continue
                stem_keys = stem_keys_func()
                if not stem_keys:
                    continue
                out_file = output_folder_mixed / model / f"({suffix_short}) {title}.{ext}"

                if enable_skip_if_file_exists and out_file.exists():
                    log(f"Skipping Mixing: {out_file})")
                    continue

                # Overlay stems
                stems_to_mix = []
                for k in stem_keys:
                    s = stems[k]
                    if name == "No Drums":
                        gain = no_drums_mix_levels.get(k, 0)
                        s = s.apply_gain(gain)
                    stems_to_mix.append(s)

                mix = stems_to_mix[0]
                for s in stems_to_mix[1:]:
                    mix = mix.overlay(s)

                mix.export(out_file, format=ext)
                if not out_file.exists() or out_file.stat().st_size < 666:
                    log(f"WARNING: Export failed (<666 bytes): {out_file.name}")
                    if out_file.exists():
                        out_file.unlink()
                    continue
                log(f"Exported: {out_file.name}")
                if enable_file_tagging and ext == "mp3":
                    set_mp3_tags(out_file, title, artist, suffix_long, suffix_short)


def delete_unmixed_stems(file: Path):
    """Delete raw separated stems after mixing."""
    for model in demucs_models:
        stem_folder = output_folder_separated / model / file.stem
        if stem_folder.exists():
            shutil.rmtree(stem_folder)
            log(f"Deleted stem folder: {stem_folder}")


def set_mp3_tags(mp3_file: Path, song_name: str, artist: str, suffix_long: str, suffix_short: str):
    """Set MP3 tags: artist, title, album, genre."""
    global track_counter
    genre = f"{artist_name} - {suffix_long}"
    try:
        audio = EasyID3(mp3_file)
    except Exception:
        audio = EasyID3()
    audio["artist"] = artist_name
    audio["title"] = song_name
    audio["album"] = f"({suffix_short}) {artist}"
    audio["date"] = str(year)
    audio["genre"] = genre
    audio["tracknumber"] = f"{track_counter:02d}"
    audio.save()
    log(f"MP3 tags set for: {mp3_file.name}")
    track_counter += 1


# --- Main Script ---
start_time = datetime.now()
log(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

mp3_files = [f for folder in input_folders for f in folder.rglob(files_filter_in)]
log(f"{len(mp3_files)} files found matching '{files_filter_in}'")
mp3_files = [f for f in mp3_files if not any(x in str(f) for x in files_filter_out)]
log(f"{len(mp3_files)} files remaining after filtering out '{files_filter_out}'")

if enable_mix:
    for model in demucs_models:
        (output_folder_mixed / model).mkdir(parents=True, exist_ok=True)

for file in tqdm(mp3_files, desc="Processing songs", colour="cyan", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
    log(f"Starting song: {file.name}")
    try:
        title, artist, album = get_source_tags(file)
        if enable_demucs:
            run_demucs(file)
        if enable_mix:
            mix_stems(file, title, artist)
        if enable_unmixed_file_deletion:
            delete_unmixed_stems(file)
    except Exception as e:
        log(f"ERROR processing {file.name}")
        log(traceback.format_exc())
        log(f"{type(e).__name__}: {e}")

finish_time = datetime.now()
duration = finish_time - start_time
log(f"Script finished at: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Duration: {duration}")

# --- Save timestamped log ---
if save_log:
    log_file_path = f"Demucs Separate Audio ({start_time.strftime('%Y-%m-%d _ %H-%M-%S')}).txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_file_output))
    print(f"Log saved: {log_file_path}")
