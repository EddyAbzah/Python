import os
import re
import subprocess
import importlib.util
from tabulate import tabulate
from datetime import date, datetime, timedelta
spec = importlib.util.spec_from_file_location("module_pixel", "Pixel Rename Files.py")
module_pixel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module_pixel)


pixel_pattern = re.compile(r"PXL_(\d{8})_(\d{9})")
whatsapp_pattern = re.compile(r".*-(\d{4})(\d{2})(\d{2})-.*")
internal_storage_path = "/storage/emulated/0/"
external_storage_path = "/sdcard/"
storage_path = internal_storage_path

min_date = date(2025, 8, 8)
jpeg_to_jpg = True
organize_into_folders = True
# organize_into_folders = False
copy_files = True
# copy_files = False


def get_full_path(path):
    path = storage_path + path
    # if path[-1] != "/":
    #     path += "/"
    return f"'{path}'"


def list_android_files(path="", folders_only=False):
    path = get_full_path(path)
    try:
        result = subprocess.run(["adb", "shell", f"ls -d {path}/*" + ("/" if folders_only else "")], capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print("ADB Error:", result.stderr)

        folders = [f.replace(storage_path, "").replace("/", "") for f in result.stdout.strip().split("\n")]
        print(f"Folders found in {path}:")
        for folder in folders:
            print("\t", folder)
    except Exception as e:
        print("Error:", e)


def list_all_files(path):
    path = get_full_path(path)
    cmd = ["adb", "shell", f"ls -l {path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print("ADB error:", result.stderr)
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
                datetime_str = (file_datetime + timedelta(seconds=i)).strftime("%Y-%m-%d _ %H-%M-%S")
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
            print(f"⚠️ ATTENTION: File already exists → {local_path}")
        else:
            remote_path = get_full_path(f"{path}/{f['filename']}")
            remote_path = remote_path.replace("'", "")
            try:
                result = subprocess.run(["adb", "pull", remote_path, local_path], capture_output=True, text=True, encoding='utf-8')
                if result.returncode == 0:
                    print(f"✅ Successfully pulled: {remote_path} -> {local_path}")
                else:
                    print(f"❌ Failed to pull: {remote_path} -> {local_path}")
            except Exception as e:
                print(f"🚨 Exception while pulling {remote_path}: {e}")


if __name__ == "__main__":
    list_android_files()

    destination_folder = r"C:\Users\eddya\Downloads\NEW"
    directories = [
                   "DCIM/Camera",
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
        filtered_files = []
        if files:
            filtered_files = [f for f in files if f['datetime'].date() >= min_date]

        print(f"\n\n\n##########   {folder_path}:   {len(filtered_files)} filtered files from {len(files)}   ##########")
        if filtered_files:
            filtered_files = get_renaming_scheme(filtered_files)
            filtered_files.sort(key=lambda x: x['datetime'])
            table_data = [f.values() for f in filtered_files]
            headers = ["Old Filename", "New Filename", "Datetime", "Size (MB)"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            if copy_files:
                if organize_into_folders:
                    if "WhatsApp" in folder_path:
                        new_destination_folder = destination_folder + "\\" + folder_path[42:].replace("/", " ")
                    else:
                        new_destination_folder = destination_folder + "\\" + os.path.basename(os.path.normpath(folder_path))
                    os.makedirs(new_destination_folder, exist_ok=True)
                else:
                    new_destination_folder = destination_folder
                pull_files(folder_path, filtered_files, new_destination_folder)
