import os
import shutil


folders_in = [
    r"C:\Users\eddya\OneDrive\תמונות\Family",
    r"C:\Users\eddya\OneDrive\תמונות\Personal",
    r"C:\Users\eddya\Videos\Jamming",
    r"C:\Users\eddya\Music",
]
folders_out = [r"C:\Users\eddya\Downloads\Files" + "\\" + folder for folder in ["Pictures - Family", "Pictures - Personal", "Videos - Jamming", "Music"]]
exclude_extensions = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".heic", ".webp",         # images
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".3gp",    # videos
    ".ffs_db"       # Other files
}

copy = False
# copy = True
print_new_folders = True
print_debug = True


def copy_files(source, destination):
    for root, _, files in os.walk(source):
        rel_path = os.path.relpath(root, source)
        dest_path = os.path.join(destination, rel_path)
        folder_files = []       # To check if a folder needs to be created not
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in exclude_extensions:
                if print_debug:
                    print(f"Skipped file: {file}")
            else:
                if print_debug:
                    print(f"Copied file: {file}")
                folder_files.append(file)

        if folder_files:
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                if print_new_folders:
                    print(f"📁 Created folder: {dest_path}")

            for file in folder_files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_path, file)
                if copy:
                    shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    for src, dst in zip(folders_in, folders_out):
        print(f"📂 Copying from {src} → {dst}")
        copy_files(src, dst)
    print("✅ All copies complete (photos & videos excluded).")
