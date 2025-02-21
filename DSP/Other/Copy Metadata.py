import os
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


def set_video_metadata(target_file, metadata):
    target_video = MP4(target_file)
    for key, value in metadata.items():
        if "xyz" in key:
            print(f'{key = }, {value = }')
            target_video[key] = value

        elif "encoded_date" in key:
            print(f'{key = }, {value = }')
            target_video["Date"] = value
    target_video.save()


directory_source = r""
directory_target = r""
true_if_video = True
copy_file_first = False
files = [f for f in os.listdir(directory_source) if f.lower().endswith(".mp4" if true_if_video else ".jpg")]
file_filter = ""


for file in files:
    if file_filter not in file:
        continue
    file_original = os.path.join(directory_source, file)
    file_new = os.path.join(directory_target, file)
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
