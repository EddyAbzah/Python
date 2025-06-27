import os
import re
from zoneinfo import ZoneInfo
from datetime import datetime, timezone


directory = r""
pattern_in = r"PXL_(\d{8})_(\d{9})"
pattern_out = "%Y-%m-%d _ %H-%M-%S"
time_zone = "Asia/Jerusalem"
# time_zone = "Asia/Tbilisi"

new_name_prefix = ""
new_name_suffix = ""
check_if_edit = [True, "~", " (edit)"]                  # enable, what to look for, suffix if found
check_if_moving_picture = [True, ".MP", " (MP)"]        # enable, what to look for, suffix if found

index_start = 1
index_count = 666666666
rename = False
# rename = True


if __name__ == '__main__':
    files = os.listdir(directory)
    files.sort()
    file_data = []
    for index, filename in enumerate(files):
        if index + 1 < index_start or index - index_start + 2 > index_count:
            continue
        match = re.match(pattern_in, filename)
        if match:
            _, file_extension = os.path.splitext(filename)
            original_utc_date = datetime.strptime(match.group(1), "%Y%m%d")
            original_utc_time = datetime.strptime(match.group(2), "%H%M%S%f").time()
            original_utc_datetime = datetime.combine(original_utc_date, original_utc_time).replace(tzinfo=timezone.utc)

            new_datetime = original_utc_datetime.astimezone(ZoneInfo(time_zone))
            new_filename = new_name_prefix
            new_filename += new_datetime.strftime(pattern_out)
            new_filename += new_name_suffix

            if check_if_edit[0] and check_if_edit[1] in filename:
                new_filename += check_if_edit[2]
            if check_if_moving_picture[0] and check_if_moving_picture[1] in filename:
                new_filename += check_if_moving_picture[2]
            if file_extension.lower() == ".jpeg":
                file_extension = ".jpg"
            new_filename += file_extension.lower()

            file_info = {"original_name": filename, "new_name": new_filename, "date": original_utc_datetime}
            file_data.append(file_info)

    existing_files = os.listdir(directory)
    duplicates = [file["new_name"] for file in file_data if file["new_name"] in existing_files]
    if duplicates:
        print(f"Duplicate files found: {', '.join(duplicates)}")
        input("Please resolve the duplicates before proceeding. Press Enter to continue.")

    file_data.sort(key=lambda x: x["date"], reverse=True)
    for file_info in file_data:
        old_path = os.path.join(directory, file_info["original_name"])
        new_path = os.path.join(directory, file_info["new_name"])
        if os.path.exists(new_path):
            print(f"ERROR! Cannot rename {file_info['original_name']} -> {file_info['new_name']}")
        else:
            if rename:
                os.rename(old_path, new_path)
                print(f"Renamed: {file_info['original_name']} -> {file_info['new_name']}")
            else:
                print(f"To be renamed: {file_info['original_name']} -> {file_info['new_name']}")
