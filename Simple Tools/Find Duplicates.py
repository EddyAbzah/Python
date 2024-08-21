import os
import re

folders = [r"", r""]
files_dict = {}  # Dictionary to store file details
originals = []  # List to store paths of duplicate files
duplicates = []  # List to store paths of duplicate files
size_up_only = False
remove_parentheses = True

# Traverse the folder
for folder_path in folders:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_size = os.path.getsize(filepath)

            if remove_parentheses:
                if '(' in file and ')' in file:
                    file = re.sub(r"\([0-9]*\)", "", file)
            if size_up_only:
                file_info = file_size
            else:
                file_info = (file, file_size)  # File name and size as key

            if file_info in files_dict:
                # Append both original and duplicate files
                if files_dict[file_info] not in originals:
                    originals.append(files_dict[file_info])  # Append the original file first
                duplicates.append(filepath)  # Append the duplicate file
            else:
                # Store file name and size
                files_dict[file_info] = filepath

# Display results
if duplicates:
    print("Original files:")
    for org in originals:
        print(org)
    print("\n\n\nDuplicate files:")
    for dup in duplicates:
        print(dup)
else:
    print("No duplicate files found.")
