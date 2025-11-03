import os
import shutil
import datetime


main_folder_out = r"C:\Users\eddya\Files\Non-Media Backup"

folders_in = [
    r"C:\Users\eddya\OneDrive\תמונות\Family",
    r"C:\Users\eddya\OneDrive\תמונות\Personal",
    r"C:\Users\eddya\Videos\Jamming",
]
folders_out = [main_folder_out + "\\" + folder for folder in ["Pictures - Family", "Pictures - Personal", "Videos - Jamming", "Music"]]
exclude_extensions = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".heic", ".webp",         # images
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp",    # videos
    ".ffs_db"                                                                   # Other files
}

music_folders_in = [r"C:\Users\eddya\Music"]
music_folders_out = [main_folder_out + "\\" + folder for folder in ["Music"]]
music_exclude_extensions = {".mp3"}

copy = False
# copy = True
print_debug = True
create_txt_file = True


txt_file_output = []


def custom_print(*args, **kwargs):
    output = ' '.join(map(str, args))
    if create_txt_file:
        txt_file_output.append(output)
    if print_debug:
        print(output)


def copy_files(source: str, destination: str, filter_extensions: set[str]):
    for root, _, files in os.walk(source):
        rel_path = os.path.relpath(root, source)
        dest_path = os.path.join(destination, rel_path)
        folder_files = []       # To check if a folder needs to be created not
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in filter_extensions:
                custom_print(f"Skipped file: {file}")
            else:
                custom_print(f"Copied file: {file}")
                folder_files.append(file)

        if folder_files:
            if len(folder_files) > 1:
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                    custom_print(f"Created folder: {dest_path}")

            for file in folder_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_path, file)
                if os.path.exists(dst_file):
                    custom_print(f"File exists: {dst_file}")
                if copy:
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        custom_print()
                        custom_print(f"ERROR!!! shutil.copy2: {e}")
                        custom_print(f'{src_file = }')
                        custom_print(f'{dst_file = }')
                        custom_print()


if __name__ == "__main__":
    # All files but pictures and videos:
    for src, dst in zip(folders_in, folders_out):
        custom_print(f"Copying from {src} → {dst}")
        copy_files(src, dst, exclude_extensions)

    # All files but mp3 files:
    for src, dst in zip(music_folders_in, music_folders_out):
        custom_print(f"Copying from {src} → {dst}")
        copy_files(src, dst, music_exclude_extensions)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d _ %H-%M-%S")
    custom_print("All copies complete")
    custom_print(f"Current time: {current_time}")

    if create_txt_file:
        filename = f"{main_folder_out}\\{current_time}.txt"
        with open(filename, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(txt_file_output))

        print(f"File '{filename}' created successfully!")
