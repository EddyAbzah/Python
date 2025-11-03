import os
import sys
import filecmp
from datetime import datetime


txt_file_output = []


def custom_print(*args, **kwargs):
    output = ' '.join(map(str, args))
    if output_text[0]:
        txt_file_output.append(output)
    print(output)


def print_list(title, list_to_print):
    custom_print(title)
    if len(list_to_print) > 0:
        for item in list_to_print:
            custom_print(item)
    else:
        custom_print('NO FILES')
    custom_print()


def are_dir_trees_equal(dir1, dir2, report_type="selective"):
    """
    Compare two directory trees content, and print all differences
    report_type: String; choose between "full", "selective", or "custom".
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if report_type == "full":
        dirs_cmp.report()
    else:
        if report_type == "selective":
            print_list("Files and subdirectories in both folders", dirs_cmp.common)
        print_list("Files and subdirectories only in folder_1:", dirs_cmp.left_only)
        print_list("Files and subdirectories only in folder_2:", dirs_cmp.right_only)
        if report_type == "selective":
            print_list("Identical in both folders, using the class’s file comparison operator:", dirs_cmp.same_files)
        print_list("Names that the type differs between the directories, or names for which os.stat() reports an error:", dirs_cmp.common_funny)
        print_list("Files which are in both folders, but could not be compared:", dirs_cmp.funny_files)

        (_, mismatch, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
        print_list("Mismatches:", mismatch)
        print_list("Errors:", errors)

        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            are_dir_trees_equal(new_dir1, new_dir2)


def are_dir_trees_equal_bool(dir1, dir2):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = filecmp.dircmp(dir1, dir2)
    if (compared.left_only or compared.right_only or compared.diff_files
            or compared.funny_files):
        return False
    for subdir in compared.common_dirs:
        if not are_dir_trees_equal_bool(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
            return False
    return True


if __name__ == "__main__":
    main_folder = r""
    output_text = [False, main_folder + "\\Compare Two Folders " + datetime.now().strftime("%H-%M-%S")]
    folder_1 = r""
    folder_2 = r""

    custom_print("\n" + "Compare Two Folders:")
    custom_print("folder_1 = " + folder_1)
    custom_print("folder_2 = " + folder_2 + "\n\n")
    are_dir_trees_equal(folder_1, folder_2)
    custom_print(f'\n-------\nSummary: {are_dir_trees_equal_bool(folder_1, folder_2) = }\n-------\n')

    if output_text[0]:
        with open(output_text[1], 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(txt_file_output))
