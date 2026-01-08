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

execute_copy = False
# execute_copy = True
delete_orphans = False
# delete_orphans = True
print_debug = True
create_txt_file = True


txt_file_output = []
had_errors = False


def custom_print(*args, **kwargs):
    output = ' '.join(map(str, args))
    if create_txt_file:
        txt_file_output.append(output)
    if print_debug:
        print(output)


def copy_files(source: str, destination: str, filter_extensions: set[str]):
    global had_errors
    for root, _, files in os.walk(source):
        rel_path = os.path.relpath(root, source)
        dest_path = os.path.join(destination, rel_path)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in filter_extensions:
                custom_print(f"Skipped file (extension): {file}")
            else:
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                    custom_print(f"Created folder: {dest_path}")

                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_path, file)
                if os.path.exists(dst_file):
                    if os.path.getmtime(src_file) <= os.path.getmtime(dst_file):
                        custom_print(f"Skipped file (up-to-date): {dst_file}")
                        continue

                custom_print(f"Copied file: {file}")
                if execute_copy:
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        had_errors = True
                        custom_print(f"\nERROR!!! shutil.copy2: {e}")
                        custom_print(f'{src_file = }')
                        custom_print(f'{dst_file = }\n')


def delete_orphaned_files(source: str, destination: str, filter_extensions: set[str]):
    global had_errors
    if not os.path.exists(destination):
        return

    for root, dirs, files in os.walk(destination, topdown=False):
        rel_path = os.path.relpath(root, destination)
        src_path = os.path.join(source, rel_path)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in filter_extensions:
                continue

            dst_file = os.path.join(root, file)
            src_file = os.path.join(src_path, file)

            if not os.path.exists(src_file):
                custom_print(f"Deleted file: {dst_file}")
                if delete_orphans:
                    try:
                        os.remove(dst_file)
                    except Exception as e:
                        had_errors = True
                        custom_print(f"\nERROR deleting file: {e}")
                        custom_print(f'{src_file = }')
                        custom_print(f'{dst_file = }\n')

        if not os.path.exists(src_path):
            if not os.listdir(root):
                custom_print(f"Deleted folder: {root}")
                if delete_orphans:
                    try:
                        os.rmdir(root)
                    except Exception as e:
                        had_errors = True
                        custom_print(f"\nERROR deleting folder: {e}")
                        custom_print(f'{src_file = }')
                        custom_print(f'{dst_file = }\n')


if __name__ == "__main__":
    # All files but pictures and videos:
    for src, dst in zip(folders_in, folders_out):
        copy_files(src, dst, exclude_extensions)
        delete_orphaned_files(src, dst, exclude_extensions)

    # All files but mp3 files:
    for src, dst in zip(music_folders_in, music_folders_out):
        copy_files(src, dst, music_exclude_extensions)
        delete_orphaned_files(src, dst, exclude_extensions)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d _ %H-%M-%S")
    if had_errors:
        custom_print("\nBackup incomplete WITH ERRORS")
    else:
        custom_print("\nBackup completed successfully with NO errors")
    custom_print(f"\nCurrent time: {current_time}")

    if create_txt_file:
        filename = f"{main_folder_out}\\{current_time}.txt"
        with open(filename, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(txt_file_output))

        print(f"File '{filename}' created successfully!")
