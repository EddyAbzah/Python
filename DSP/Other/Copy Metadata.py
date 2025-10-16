import os
import re
from pymediainfo import MediaInfo
from mutagen.mp4 import MP4
import shutil
import piexif


def get_all_metadata(file_path):
    media_info = MediaInfo.parse(file_path)
    metadata = {}
    for track in media_info.tracks:
        if track.track_type == "Video" or track.track_type == "General":
            for key, value in track.to_data().items():
                metadata[key] = value
    return metadata


def convert_coordinates(coordinate_str):
    pattern = r'([+-]?\d{1,2}\.\d+)([+-]?\d{1,3}\.\d+)'
    if match := re.match(pattern, coordinate_str):
        lat, lon = match.groups()
        lat = float(lat)
        lon = float(lon)
        lat_direction = 'N' if lat > 0 else 'S'
        lon_direction = 'E' if lon > 0 else 'W'
        lat = abs(lat)
        lon = abs(lon)
        return f"{lat:.2f}°{lat_direction} {lon:.2f}°{lon_direction}"
    else:
        return coordinate_str


def set_video_metadata(target_file, metadata):
    target_video = MP4(target_file)
    for key, value in metadata.items():
        if "xyz" in key:
            print(f'{key = }, {value = }')
            target_video[key] = value
            target_video["@LOC"] = convert_coordinates(value)

        elif "encoded_date" in key:
            print(f'{key = }, {value = }')
            target_video["Date"] = value
    target_video.save()


directory_source = r"C:\Users\eddya\Downloads\Pixel Media"
directory_target = r"C:\Users\eddya\Videos"
true_if_video = True
copy_file_first = False
file_filter = ""

files = [os.path.join(root, f) for root, _, filenames in os.walk(directory_source) for f in filenames if f.lower().endswith(".mp4" if true_if_video else ".jpg")]
# files = [os.path.join(directory_source, f) for f in os.listdir(directory_source) if f.lower().endswith(".mp4" if true_if_video else ".jpg")]


for file_original in files:
    filename = os.path.basename(file_original)
    if file_filter not in filename:
        continue
    file_new = os.path.join(directory_target, filename)
    if copy_file_first:
        shutil.copy(file_original, file_new)
    try:
        if true_if_video:
            metadata = get_all_metadata(file_original)
            if metadata:
                set_video_metadata(file_new, metadata)
        else:
            exif_data = piexif.load(file_original)
            exif_bytes = piexif.dump(exif_data)
            piexif.insert(exif_bytes, file_new)
    except:
        print(f"File {file_new} not found")
    else:
        print(f"All EXIF data copied from {file_original} to {file_new}.")
