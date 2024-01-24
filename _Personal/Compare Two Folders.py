from datetime import datetime
# import keyboard
import filecmp
import os


text = [False, [], os.path.dirname(__file__) + "\\Compare Two Folders " + datetime.now().strftime("%H-%M-%S")]
folder_1 = r"C:\Users\eddy.a\OneDrive - SolarEdge\Documents\AutomationConsole Scripts_Git\Infrastructure\PLC\_TEMP\Arc - EDIT"
folder_2 = r"C:\Users\eddy.a\OneDrive - SolarEdge\Documents\AutomationConsole Scripts_Git\Infrastructure\PLC\_TEMP\Arc"


def main():
    print_item("Compare Two Folders:")
    print_item("folder_1 = " + folder_1)
    print_item("folder_2 = " + folder_2)
    print_item()
    print_item()
    are_dir_trees_equal(folder_1, folder_2)
    finish()


def finish():
    if text[0]:
        with open(f"{text[2]}.txt", 'w', encoding="utf-8") as f:
            f.write("\n".join(text[1]))


def print_list(title, df):
    if len(df) > 0:
        print_item(title)
        for data in df:
            print_item(data)
        print_item()


def are_dir_trees_equal(dir1, dir2):
    # if keyboard.is_pressed('q'):
    #     finish()
    # else:
        print_item("###################################################################################")
        print_item("Current folder = " + dir1.split('\\')[-1])
        dirs_cmp = filecmp.dircmp(dir1, dir2)
        print_list("Funny files - folder_1:", dirs_cmp.left_only)
        print_list("Funny files - folder_2:", dirs_cmp.right_only)
        print_list("Funny files:", dirs_cmp.funny_files)

        (_, mismatch, errors) = filecmp.cmpfiles(dir1, dir2, dirs_cmp.common_files, shallow=False)
        print_list("Mismatches:", mismatch)
        print_list("Errors:", errors)

        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            are_dir_trees_equal(new_dir1, new_dir2)
        print_item()


def print_item(string=""):
    if text[0]:
        text[1].append(string)
    print(string)


if __name__ == "__main__":
    main()
