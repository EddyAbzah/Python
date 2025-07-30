import os
import re
import piexif
import subprocess
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from pymediainfo import MediaInfo
from datetime import datetime, timedelta


def write_creation_date_jpg(path, dt):
    try:
        exif_dict = piexif.load(path)
        date_str = dt.strftime("%Y:%m:%d %H:%M:%S").encode()

        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_str
        exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = date_str
        exif_dict['0th'][piexif.ImageIFD.DateTime] = date_str

        try:
            exif_bytes = piexif.dump(exif_dict)
        except ValueError as e:
            match = re.search(r'"dump" got wrong type of exif value\.\s*(\d+)', str(e))
            if match:
                tag = int(match.group(1))
                print(f"EXIF tag {tag} has an invalid type. Removing it.")
                exif_dict['Exif'].pop(tag, None)
                exif_bytes = piexif.dump(exif_dict)
            else:
                raise  # it's a different ValueError

        piexif.insert(exif_bytes, path)
        return True

    except Exception as e:
        print(f"Failed to write JPG EXIF to {path}: {e}")
        return False


def write_creation_date_mp4(file_path, dt):
    iso_dt = dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # must be UTC ISO format
    temp_file = file_path.replace(".mp4", "_temp.mp4")

    try:
        result = subprocess.run(["ffmpeg", "-y", "-i", file_path, "-metadata", f"creation_time={iso_dt}", "-codec", "copy", temp_file],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        os.replace(temp_file, file_path)
        return True

    except subprocess.CalledProcessError as e:
        print(f"[ffmpeg error] {file_path}:\n{e.stderr}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False


def main():
    results = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.jpg')):
                full_path = os.path.join(subdir, file)
                file_name = os.path.basename(full_path)
                new_file_name = None
                metadata_dt = None
                new_metadata = None

                if file.lower().endswith('.mp4'):
                    media_info = MediaInfo.parse(full_path)
                    for track in media_info.tracks:
                        if track.track_type in ("Video", "General"):
                            for key, value in track.to_data().items():
                                if "encoded_date" in key and isinstance(value, str):
                                    try:
                                        value = re.sub(r'\s*UTC\s*', '', value)
                                        metadata_dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                                    except ValueError:
                                        pass
                                    break
                else:  # .jpg
                    try:
                        with Image.open(full_path) as img:
                            exif_data = img._getexif()
                            if exif_data:
                                for tag_id, value in exif_data.items():
                                    tag = TAGS.get(tag_id, tag_id)
                                    if tag == 'DateTimeOriginal':
                                        metadata_dt = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                                        break
                    except Exception:
                        pass

                if metadata_dt:
                    metadata_str = metadata_dt.strftime('%Y-%m-%d %H-%M-%S')
                else:
                    metadata_str = "N/A"

                # --- Parse date from filename ---
                match_format = "No date"
                match_result = False
                file_datetime = None

                if full_match := full_pattern.search(file_name):
                    match_format = "Full"
                    date_str = f"{full_match.group(1)} {full_match.group(2).replace('-', ':')}"
                    try:
                        file_datetime = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass
                elif date_match := date_only_pattern.search(file_name):
                    match_format = "Date only"
                    date_str = date_match.group(1)
                    try:
                        file_datetime = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        pass
                elif date_match := whatsapp_pattern.search(file_name):
                    match_format = "Date only"
                    date_str = date_match.group(1)
                    try:
                        file_datetime = datetime.strptime(date_str, '%Y%m%d')
                    except ValueError:
                        pass

                # --- Compare metadata and filename ---
                edit_method = "No adjustments"
                if file_datetime and metadata_dt:
                    if match_format == "Full":
                        match_result = abs(metadata_dt - file_datetime) <= max_offset
                        if not match_result and abs(metadata_dt - file_datetime) <= rename_file_from_metadata[1]:
                            edit_method = "Rename file"
                    elif match_format == "Date only":
                        match_result = metadata_dt.strftime('%Y-%m-%d') == file_datetime.strftime('%Y-%m-%d')
                if not file_datetime and metadata_dt:
                    edit_method = "Rename file"
                elif not match_result and edit_method != "Rename file" and match_format != "No date":
                    edit_method = "Edit metadata"

                if edit_method == "Edit metadata":
                    new_metadata = file_datetime.strftime("%Y-%m-%d %H-%M-%S")
                    if edit_files and write_metadata_from_filename:
                        if file.lower().endswith('.jpg'):
                            success = write_creation_date_jpg(full_path, file_datetime)
                            if success:
                                print(f'Edit EXIF metadata\t{full_path}\t{file_datetime}')
                        elif file.lower().endswith('.mp4'):
                            success = write_creation_date_mp4(full_path, file_datetime)
                            if success:
                                print(f'Edit MP4 metadata\t{full_path}\t{file_datetime}')

                elif edit_method == "Rename file":
                    basename, extension = os.path.splitext(file)
                    new_file_name = metadata_dt.strftime('%Y-%m-%d _ %H-%M-%S') + extension
                    if edit_files and rename_file_from_metadata[0]:
                        new_full_path = os.path.join(subdir, new_file_name)
                        os.rename(full_path, new_full_path)
                        print(f'Rename file\t{full_path}\t{new_full_path}')

                # --- Append result row ---
                results.append({'File path': subdir, 'Old file name': file_name, 'New file name': new_file_name, 'Extension': file_name[-4:], 'Date format': match_format,
                                'Old metadata': metadata_str, 'New metadata': new_metadata, 'Match': match_result, 'Edit method': edit_method})
    return results


if __name__ == '__main__':
    root_dir = r"C:\Users\eddya\OneDrive\תמונות\Family"
    max_offset = timedelta(minutes=10)
    edit_files = True
    write_metadata_from_filename = True
    rename_file_from_metadata = [True, timedelta(hours=4)]

    full_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})\s*[_\-]\s*(\d{2}-\d{2}-\d{2})')
    date_only_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    whatsapp_pattern = re.compile(r'-(\d{4}\d{2}\d{2})-WA')

    df = pd.DataFrame(main())
    df.to_excel(r"C:\Users\eddya\Downloads\MANA.xlsx", index=False)
