import os
import re
from collections import defaultdict


folder_path = r""
rename = True
jpeg_to_jpg = True

pattern = re.compile(r".*-(\d{4})(\d{2})(\d{2})-.*")
date_counters = defaultdict(int)
files = sorted(os.listdir(folder_path))

for filename in files:
    match = pattern.search(filename)
    if match:
        year, month, day = match.groups()
        formatted_date = f"{year}-{month}-{day}"

        date_counters[formatted_date] += 1
        count = date_counters[formatted_date]

        ext = os.path.splitext(filename)[1].lower()
        if jpeg_to_jpg and ext == "jpeg":
            ext = "jpg"
        new_name = f"{formatted_date} _ {count:03}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        if rename:
            os.rename(src, dst)
        print(f"Renamed: {filename} → {new_name}")
    else:
        print(f"Skipped (no date match): {filename}")

print("✅ Done renaming files.")
