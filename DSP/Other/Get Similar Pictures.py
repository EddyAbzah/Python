import os
import shutil
import imagehash
from PIL import Image
from collections import defaultdict


log_messages = []


def log(message):
    print(message)
    log_messages.append(message)


def find_and_copy_duplicates():
    os.makedirs(output_directory, exist_ok=True)
    hashes = defaultdict(list)

    for directory in input_directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    filepath = os.path.join(root, file)
                    try:
                        with Image.open(filepath) as img:
                            img_hash = str(hash_function(img))
                            hashes[img_hash].append(filepath)
                    except Exception as e:
                        print(f"Could not process {filepath}: {e}")

    group_num = 1
    duplicates_found = False
    for hash_value, files in hashes.items():
        if len(files) > 1:
            duplicates_found = True
            os.makedirs(output_directory, exist_ok=True)
            log(f"\n{group_num:03} - Duplicate images with hash {hash_value}:")
            for i, file in enumerate(files, start=1):
                log(file)
                _, ext = os.path.splitext(file)
                new_filename = f"{group_num:03} - {i:02} _ {os.path.basename(file)}"
                dest_path = os.path.join(output_directory, new_filename)
                shutil.copy2(file, dest_path)
            group_num += 1
    if not duplicates_found:
        print("No duplicate images found.")


if __name__ == "__main__":
    input_directories = [r"", r""]
    output_directory = r""

    hash_function = imagehash.phash
    # hash_function = imagehash.dhash
    hash_threshold = 5  # Lower = stricter (0 = exact match)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    find_and_copy_duplicates()
    with open(output_directory + r"\Get Similar Pictures.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_messages))
