import os
import cv2
import numpy as np
from fnmatch import fnmatch
from datetime import datetime
from functools import partial

# Files
folders_in = [r"C:\Users\eddy.a\Downloads"]
include_subfolders = True
filter_in = ["*.jpg"]
filter_out = [""]
folder_out = ""
edit_name = "edit 01"
timestamp = "%Y-%m-%d %H-%M-%S"

# Conversion
Method = ["Gray-world algorithm", "White patch reference"][1]
open_photo_before_edit = True
open_photo_after_edit = True
open_photo_type = [cv2.WINDOW_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED][0]
White_patch = {"x_start": 0.4, "y_start": 0.4, "x_stop": 0.6, "y_stop": 0.6}


def get_files(folders_in, include_subfolders, filter_in, filter_out):
    """
    Get all files, with a matching names using fnmatch.fnmatch().
    Args:
        folders_in: List of folders / Strings.
        include_subfolders: Bool.
        filter_in: List of Strings; if you leave this empty, you will be left with [''].
        filter_out: List of Strings; if you leave this empty, you will be left with [''].
    """
    files = []
    for folder in folders_in:
        folder = folder.strip()
        for root, dirs, all_files in os.walk(folder):
            for file in all_files:
                if (filter_in[0] == "" or any(fnmatch(file, f) for f in filter_in)) and (filter_out[0] == "" or not any(fnmatch(file, f) for f in filter_out)):
                    files.append(os.path.join(root, file))
            if not include_subfolders:
                break
    return files

def edit_photo(path, file_out):
    global mouse_click_counter
    mouse_click_counter = 0
    # reading the image; Flags=1: Any transparency of image will not be neglected
    img = cv2.imread(path, flags=None if Method == "Gray-world algorithm" else 1)
    if Method == "White patch reference":
        show_image(img.copy(), f"{path} - pick the white spot", with_mouse_callback=True)

    if Method == "Gray-world algorithm":
        img_edit = gray_world_white_balance(img)
    else:
        img_edit = white_patch_reference(img)

    if open_photo_before_edit:
        show_image(img, f"{path} - before edit")
    if open_photo_after_edit:
        show_image(img_edit, f"{path} - after edit")
    if Method == "White patch reference":
        # Convert to 8 bit before saving
        img_edit = (img_edit * 255).astype(int)
    cv2.imwrite(file_out, img_edit)


def gray_world_white_balance(image, brightness_factor=1.0):
    """
    It assumes that average pixel value is neutral gray (128) because of good distribution of colors.
    So we can estimate pixel color by looking at the average color.
    """
    # We will convert the image to LAB color space: L for lightness, A for Red/Green and B for Blue/Yellow
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # We will calculate the mean color values in A and B channels
    avg_a = np.average(img_lab[:, :, 1])
    avg_b = np.average(img_lab[:, :, 2])
    # Then, subtract 128 (mid-gray) from the means and normalize the L channel by multiplying with this difference; and finally, subtract this value from A and B channels
    img_lab[:, :, 1] = img_lab[:, :, 1] - ((avg_a - 128) * (img_lab[:, :, 0] / 255.0) * brightness_factor)
    img_lab[:, :, 2] = img_lab[:, :, 2] - ((avg_b - 128) * (img_lab[:, :, 0] / 255.0) * brightness_factor)
    balanced_image = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return balanced_image


def white_patch_reference(image):
    """
    Pick a patch of image that is supposed to be white to use it as reference to rescale each channel in the image
    Please note that we will get the w pixl value first and h next from the above exercise
    If you consider the image shape to be h x w x 3
    """
    x_axis = sorted([White_patch["x_start"], White_patch["x_stop"]])
    y_axis = sorted([White_patch["y_start"], White_patch["y_stop"]])
    image_patch = image[y_axis[0]:y_axis[1], x_axis[0]:x_axis[1]]
    # Get maximum pixel values from each channel (BGR), normalize the original image
    # with these max pixel values - assuming the max pixel is white.
    image_normalized = image / image_patch.max(axis=(0, 1))
    # Some values will be above 1, so we need to clip the values to between 0 and 1
    image_balanced = image_normalized.clip(0, 1)    # to display
    cv2.rectangle(image, (x_axis[0], y_axis[0]), (x_axis[1], y_axis[1]), (0, 0, 255), 2)
    return image_balanced


def mouse_click_event(image, window_name, event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = image[y, x, 0]
        g = image[y, x, 1]
        r = image[y, x, 2]
        font_color = int(abs(255 - b)), int(abs(255 - g)), int(abs(255 - r))
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouse_click_counter
        global White_patch
        if mouse_click_counter == 0:
            White_patch["x_start"] = x
            White_patch["y_start"] = y
        elif mouse_click_counter == 1:
            White_patch["x_stop"] = x
            White_patch["y_stop"] = y
        else:
            White_patch["x_start"] = White_patch["x_stop"]
            White_patch["y_start"] = White_patch["y_stop"]
            White_patch["x_stop"] = x
            White_patch["y_stop"] = y
        cv2.putText(image, f'{x=}, {y=}', (x, y), font, 2, font_color, 2)
        cv2.imshow(window_name, image)
        mouse_click_counter += 1
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.putText(image, f'{r=}, {g=}, {b=}', (x, y), font, 2, font_color, 2)
        cv2.imshow(window_name, image)


def show_image(image, window_name, with_mouse_callback=False):
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, open_photo_type)
    cv2.imshow(window_name, image)
    if with_mouse_callback:
        global White_patch
        White_patch["x_start"] = int(White_patch["x_start"] * image.shape[1])
        White_patch["y_start"] = int(White_patch["y_start"] * image.shape[0])
        White_patch["x_stop"] = int(White_patch["x_stop"] * image.shape[1])
        White_patch["y_stop"] = int(White_patch["y_stop"] * image.shape[0])
        cv2.setMouseCallback(window_name, partial(mouse_click_event, image, window_name))
    cv2.waitKey(0)
    # Original, but this is slower:
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    mouse_click_counter = 0
    files = get_files(folders_in, include_subfolders, filter_in, filter_out)
    if len(files) > 0:
        if folder_out != "":
            os.makedirs(folder_out, exist_ok=True)
        for index_file, file in enumerate(files):
            file_out = file
            if edit_name != "":
                file_out = f" ({edit_name}).".join(file.rsplit('.', 1))
            if timestamp != "":
                file_out = f" _ {timestamp})".join(file_out.rsplit(')', 1))
            if folder_out != "":
                file_out = folder_out + "\\" + file_out.rsplit('\\', 1)[-1]
            edit_photo(file, file_out)
    else:
        raise Exception("There are no files to edit")
