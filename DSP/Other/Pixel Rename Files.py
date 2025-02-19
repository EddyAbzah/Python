import os
import re
from datetime import datetime, timedelta


directory = r""
pattern_in = r"PXL_(\d{8})_(\d{9})"
pattern_out = "%Y-%m-%d _ %H-%M-%S"
new_name_prefix = ""
new_name_suffix = "_PXL"
time_delta = timedelta(days=0, hours=2, seconds=0, minutes=0)
file_data = []

index_start = 1
index_count = 666666
rename = False
# rename = True


files = os.listdir(directory)
files.sort()
for index, filename in enumerate(files):
    if index + 1 < index_start or index - index_start + 2 > index_count:
        continue
    match = re.match(pattern_in, filename)
    if match:
        _, file_extension = os.path.splitext(filename)
        original_date = datetime.strptime(match.group(1), "%Y%m%d")
        original_time = datetime.strptime(match.group(2), "%H%M%S%f").time()
        original_datetime = datetime.combine(original_date, original_time)

        new_datetime = original_datetime + time_delta
        new_filename = new_name_prefix
        new_filename += new_datetime.strftime(pattern_out)
        new_filename += new_name_suffix
        new_filename += file_extension.lower()

        file_info = {"original_name": filename, "new_name": new_filename, "date": original_datetime}
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
