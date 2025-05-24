import os
import ctypes
from tabulate import tabulate


def get_file_attributes(path):
    file_attribute_hidden = 0x2
    file_attribute_system = 0x4
    file_attribute_readonly = 0x1

    attributes = ctypes.windll.kernel32.GetFileAttributesW(path)
    if attributes == -1:
        return None  # Unable to retrieve attributes

    hidden = bool(attributes & file_attribute_hidden)
    system = bool(attributes & file_attribute_system)
    readonly = bool(attributes & file_attribute_readonly)

    return hidden, system, readonly


def list_protected_files(directory):
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    data = []
    for root, dirs, files in os.walk(directory):
        for item in dirs + files:
            full_path = os.path.join(root, item)
            file_name = os.path.basename(full_path)
            file_path = os.path.dirname(full_path)
            attributes = get_file_attributes(full_path)

            if attributes:
                hidden, system, readonly = attributes
                if hidden or system or readonly:
                    data.append([file_name, file_path, hidden, system, readonly])

    if data:
        print(tabulate(data, headers=["File Name", "Path", "Hidden", "System", "Read-Only"], tablefmt="grid"))
    else:
        print("No protected files found.")


if __name__ == "__main__":
    directory = r""
    list_protected_files(directory)
