"""
Find all files that have the same date in their filenames but are located in different folders.
This script recursively scans directories, extracts dates from filenames, and finds those dates that appear in files from different directories.
"""


import os
import re
from collections import defaultdict
from tabulate import tabulate


root_dir = r""
date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}) _ .*')

date_to_files = defaultdict(list)
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        match = date_pattern.match(filename)
        if match:
            date = match.group(1)
            full_path = os.path.join(dirpath, filename)
            date_to_files[date].append(full_path)

table = []
for date, files in date_to_files.items():
    folders = {os.path.dirname(f) for f in files}
    if len(folders) > 1:
        file_list = '\n'.join(files)
        table.append([date, file_list])

if table:
    print(tabulate(table, headers=["Date", "Files"], tablefmt="fancy_grid"))
else:
    print("No matching dates found across different folders.")
