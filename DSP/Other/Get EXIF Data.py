import os
import pytz
import shutil
from PIL import Image
from PIL.ExifTags import TAGS
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from datetime import datetime, timedelta


def extract_exif_data(image_path):
    """Extract the EXIF date and device maker (manufacturer) from an image file."""
    exif_date, device_maker = None, None
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                decoded_tag = TAGS.get(tag, tag)
                if decoded_tag == 'DateTimeOriginal':
                    # EXIF Date format is "YYYY:MM:DD HH:MM:SS"
                    exif_date = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                if decoded_tag == 'Make':
                    device_maker = value.strip().replace(" ", "_")

    except Exception as e:
        print(f"{image_path}: Error reading EXIF because {e}")
    return exif_date, device_maker


def extract_metadata(video_path):
    """Get creation date from MP4 file using hachoir and adjust to Jerusalem time."""
    exif_date, device_maker = None, "Video"
    try:
        # Create a parser for the file
        parser = createParser(video_path)
        metadata = extractMetadata(parser)

        if metadata:
            # Extract creation date in UTC
            creation_date = metadata.get("creation_date")
            if creation_date:
                # Ensure it's a datetime object (for safety)
                if isinstance(creation_date, str):
                    creation_date = datetime.strptime(creation_date, '%Y-%m-%d %H:%M:%S')
                if creation_date.year > 2005:   # my digital camera defaults to this year
                    # Define UTC and Jerusalem timezones
                    utc_timezone = pytz.utc
                    jerusalem_timezone = pytz.timezone("Asia/Jerusalem")
                    # Convert the creation date from UTC to Jerusalem time
                    creation_date_utc = utc_timezone.localize(creation_date)
                    exif_date = creation_date_utc.astimezone(jerusalem_timezone)
        parser.close()
        return exif_date, device_maker

    except Exception as e:
        print(f"Error extracting creation date from {video_path}: {e}")
        return exif_date, device_maker


def rename_file(image_path, new_name):
    """Rename the image file to the new name."""
    folder = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1]
    new_file_path = os.path.join(folder, f"{new_name}{ext}")
    if image_path == new_file_path:
        print(f"{image_path}: File name not changed.")
    else:
        if not os.path.exists(new_file_path):
            shutil.move(image_path, new_file_path)
            print(f"{image_path}: Renamed to\n{new_file_path}")
        else:
            print(f"{image_path}: File already exists.")
    print()


if __name__ == "__main__":
    folder_path = r""
    include_subfolders = True
    device_hour_offset = ["", 1]    # Look for the device and add or subtract time for time zones
    image_formats = ('.jpg', '.jpeg')
    video_formats = ('.mp4', )
    output_date_format = '%Y-%m-%d _ %H-%M-%S'  # Format the date as 'YYYY-MM-DD_HH-MM-SS'
    keep_original_name = False
    number_the_files = [True, 1]
    rename_if_no_exif_data = True

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_formats + video_formats):
                file_path = os.path.join(root, file)
                if file.lower().endswith(image_formats):
                    exif_date, device_maker = extract_exif_data(file_path)
                else:
                    exif_date, device_maker = extract_metadata(file_path)
                if exif_date:
                    if device_maker == device_hour_offset[0]:
                        # Adjust the EXIF date by the specified hour offset
                        exif_date = exif_date + timedelta(hours=device_hour_offset[1])

                    if keep_original_name:
                        new_name = os.path.splitext(file_path)[0] + " _ " + exif_date.strftime(output_date_format)
                    else:
                        new_name = exif_date.strftime(output_date_format)
                    if number_the_files[0]:
                        new_name = f"{number_the_files[1]:03} _ {new_name}"
                    rename_file(file_path, new_name)
                else:
                    if rename_if_no_exif_data:
                        if number_the_files[0]:
                            new_name = f"{number_the_files[1]:03} _ no EXIF data"
                        else:
                            new_name = os.path.splitext(file_path)[0] + " (no EXIF data)"
                        rename_file(file_path, new_name)
                    else:
                        print(f"{file_path}: No EXIF data found!")
            number_the_files[1] += 1

        if not include_subfolders:
            break  # If we don't want to include subfolders, stop after the first iteration
