"""Simple script to renames WhatsApp media by re-numbering"""


import os
import re


def renumber_files_in_folder():
    files = [f for f in os.listdir(folder_path) if pattern.search(f)]
    print(f'{len(files) = }\n')
    files.sort()

    last_date = "0000-00-00"
    counter = 1
    before_names = []
    after_names = []

    # Go through the files and renumber
    for filename in files:
        date = filename[:10]
        if date == last_date:
            counter += 1
        else:
            counter = 1
        last_date = date

        new_filename = f"{date} _ {counter:03}" + filename[16:]
        if renamed_suffixed:
            new_filename = new_filename.replace("_.", ".")
        if new_filename != filename:
            before_names.append(filename)
            after_names.append(new_filename)

    before_names.reverse()
    after_names.reverse()
    for before, after in zip(before_names, after_names):
        old_filepath = os.path.join(folder_path, before)
        new_filepath = os.path.join(folder_path, after)
        try:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {before} -> {after}")
        except:
            new_after = after[:-4] + "_" + after[-4:]
            new_filepath = os.path.join(folder_path, new_after)
            os.rename(old_filepath, new_filepath)
            print(f"Suffixed: {before} -> {new_after}")


pattern = re.compile(r'\d{4}-\d{2}-\d{2} _ \d{3}.*')
renamed_suffixed = False
folder_path = r""

renumber_files_in_folder()
