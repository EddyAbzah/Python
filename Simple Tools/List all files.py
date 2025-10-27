import os
import csv


main_folder_out = r"C:\Users\eddya\Files\_Sync\All files"
folders = [
    (r"C:\Users\eddya\Files", "Files - Files"),
    (r"C:\Users\eddya\Music", "Files - Music"),
    (r"C:\Users\eddya\Videos\Jamming", "Videos - Jamming"),
    (r"C:\Users\eddya\OneDrive\תמונות\Family", "Pictures - Family"),
    (r"C:\Users\eddya\OneDrive\תמונות\Personal", "Pictures - Personal"),
    (r"C:\Users\eddya\OneDrive\מסמכים\Python", "Python"),
    (r"C:\Users\eddya\Videos\Guitar Covers", "Videos - Guitar Covers"),
    (r"C:\Users\eddya\Downloads", "Files - Downloads"),
    (r"C:\Users\eddya\Documents", "Files - Documents"),
]

create_txt_file = True
txt_file_output = []


def custom_print(*args, **kwargs):
    output = ' '.join(map(str, args))
    if create_txt_file:
        txt_file_output.append(output)
    print(output)


def list_files_to_csv(root_folder, output_csv):
    headers = ['Path', 'Filename', 'Extension', 'Size']

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                size_bytes = os.path.getsize(full_path)
                size_mb = round(size_bytes / (1024 * 1024), 3)  # MB, rounded to 3 decimals
                _, extension = os.path.splitext(filename)

                writer.writerow([dirpath, filename, extension, size_mb])

    print(f"✅ File list written to: {output_csv}")


if __name__ == "__main__":
    for folder_in, file_out in folders:
        output_csv = main_folder_out + "\\" + file_out + ".csv"
        list_files_to_csv(folder_in, output_csv)
