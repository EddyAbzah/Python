import os
import re
import subprocess
import importlib.util
from tabulate import tabulate
from datetime import datetime, timedelta
spec = importlib.util.spec_from_file_location("module_pixel", "Pixel Rename Files.py")
module_pixel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_pixel)


pixel_pattern = re.compile(r"PXL_(\d{8})_(\d{9})")
whatsapp_pattern = re.compile(r".*-(\d{4})(\d{2})(\d{2})-.*")
internal_storage_path = "/storage/emulated/0/"
external_storage_path = "/sdcard/"
storage_path = internal_storage_path
date_format = "%Y-%m-%d _ %H-%M-%S"

min_date = [True, True, datetime(2026, 4, 1, 0, 0)]       # Enable, Check External file, Manual min_time
jpeg_to_jpg = True
organize_into_folders = True
# organize_into_folders = False
copy_files = True
# copy_files = False
create_txt_file = True


txt_file_output = []


def custom_print(*args, **kwargs):
    output = ' '.join(map(str, args))
    if create_txt_file:
        txt_file_output.append(output)
    print(output)


def get_full_path(path):
    path = storage_path + path
    # if path[-1] != "/":
    #     path += "/"
    return f"'{path}'"


def get_latest_timestamp(destination_folder):
    latest_date = None
    if min_date[1]:
        for filename in os.listdir(destination_folder):
            if not filename.endswith(".txt"):
                continue
            name_without_ext = filename[:-4]
            try:
                dt = datetime.strptime(name_without_ext, date_format)
                if latest_date is None or dt > latest_date:
                    latest_date = dt
            except ValueError:
                continue

    if not min_date[1] or not latest_date:
        latest_date = min_date[2]
    custom_print(f"\n\n\n##########   Filter files from = {latest_date.strftime(date_format)}   ##########")
    return latest_date


def list_android_files(path="", folders_only=False):
    path = get_full_path(path)
    try:
        result = subprocess.run(["adb", "shell", f"ls -d {path}/*" + ("/" if folders_only else "")], capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            custom_print("ADB Error:", result.stderr)

        folders = [f.replace(storage_path, "").replace("/", "") for f in result.stdout.strip().split("\n")]
        custom_print(f"Folders found in {path}:")
        for folder in folders:
            custom_print("\t", folder)
    except Exception as e:
        custom_print("Error:", e)


def list_all_files(path):
    path = get_full_path(path)
    cmd = ["adb", "shell", f"ls -l {path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        custom_print("ADB error:", result.stderr)
        return []

    files = []
    for line in result.stdout.strip().split('\n'):
        if not line or line.startswith('total'):
            continue
        match = re.match(r'^(\S+)\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(.+)$', line)
        if not match:
            continue
        permissions, size_str, date_str, time_str, filename = match.groups()
        if permissions.startswith('d'):     # Skip directories
            continue

        size = round(int(size_str) / (1024 * 1024), 2)
        file_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        files.append({"filename": filename, "datetime": file_datetime, "size": size})
    return files


def get_renaming_scheme(files_in):
    files = []
    for file in files_in:
        filename, file_datetime, size = file.values()

        if whatsapp_pattern.search(filename):
            _, file_extension = os.path.splitext(filename)
            for i in range(0, 60):
                datetime_str = (file_datetime + timedelta(seconds=i)).strftime(date_format)
                filename_new = datetime_str + file_extension
                if filename_new not in [d["filename_new"] for d in files if "filename_new" in d]:
                    break
        elif pixel_pattern.search(filename):
            filename_new = module_pixel.get_renaming_scheme(filename)["new_name"]
        else:
            filename_new = filename
        if jpeg_to_jpg:
            filename_new.replace(".jpeg", ".jpg")
        files.append({"filename": filename, "filename_new": filename_new, "datetime": file_datetime, "size": size})
    return files


def pull_files(path, files, dest_folder):
    for f in files:
        local_path = f"{dest_folder}\\{f['filename_new']}"
        if os.path.exists(local_path):
            custom_print(f"ATTENTION: File already exists → {local_path}")
        else:
            remote_path = get_full_path(f"{path}/{f['filename']}")
            remote_path = remote_path.replace("'", "")
            try:
                result = subprocess.run(["adb", "pull", remote_path, local_path], capture_output=True, text=True, encoding='utf-8')
                if result.returncode == 0:
                    custom_print(f"Successfully pulled: {remote_path} -> {local_path}")
                else:
                    custom_print(f"Failed to pull: {remote_path} -> {local_path}")
            except Exception as e:
                custom_print(f"Exception while pulling {remote_path}: {e}")


if __name__ == "__main__":
    list_android_files()

    destination_folder = r"C:\Users\eddya\Downloads\Pixel Media"
    directories = [
                   "DCIM/Camera",
                   "DCIM/Blackmagic Camera",
                   "DCIM/CapCut",
                   "DCIM/CapCut",
                   "DCIM/Insta360Download",
                   "Android/data/com.arashivision.insta360akiko/files/Insta360OneR/galleryOriginal/Camera01",
                   "Android/data/com.arashivision.insta360akiko/files/Insta360OneR/galleryOriginal/LRV",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Animated Gifs",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Animated Gifs/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Audio/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Documents",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Documents/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Images/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Stickers",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Stickers/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video Notes",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Video Notes/Sent",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes",
                   "Android/media/com.whatsapp/WhatsApp/Media/WhatsApp Voice Notes/Sent",
                   ]
    for folder_path in directories:
        files = list_all_files(folder_path)
        custom_print(f"\n\n\n##########   {folder_path}:   Number of files = {len(files)}   ##########")
        if files and min_date[0]:
            latest_timestamp = get_latest_timestamp(destination_folder)
            files = [f for f in files if f['datetime'] >= latest_timestamp]
            custom_print(f"\n\n\n##########   {folder_path}:   Files after filter = {len(files)}   ##########")

        if files:
            files = get_renaming_scheme(files)
            files.sort(key=lambda x: x['datetime'])
            table_data = [f.values() for f in files]
            headers = ["Old Filename", "New Filename", "Datetime", "Size (MB)"]
            custom_print(tabulate(table_data, headers=headers, tablefmt="grid"))

            if copy_files:
                if organize_into_folders:
                    if "WhatsApp" in folder_path:
                        new_destination_folder = destination_folder + "\\" + folder_path[42:].replace("/", " ")
                    else:
                        new_destination_folder = destination_folder + "\\" + os.path.basename(os.path.normpath(folder_path))
                    os.makedirs(new_destination_folder, exist_ok=True)
                else:
                    new_destination_folder = destination_folder
                pull_files(folder_path, files, new_destination_folder)

    current_time = datetime.now().strftime(date_format)
    custom_print("All copies complete")
    custom_print(f"Current time: {current_time}")

    if create_txt_file:
        filename = f"{destination_folder}\\{current_time}.txt"
        with open(filename, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(txt_file_output))

        print(f"File '{filename}' created successfully!")
