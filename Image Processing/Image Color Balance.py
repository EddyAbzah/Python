import cv2
import numpy as np


Method = ["Gray-world algorithm", "White patch reference"][1]
path = r"C:\Users\eddy.a\Downloads\MANA\Pre 003.jpg"
open_photo_before_edit = True
open_photo_after_edit = True
open_photo_type = [cv2.WINDOW_FULLSCREEN, cv2.WINDOW_GUI_EXPANDED][0]


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
    # Defining a small rectangle
    h_start, w_start, h_width, w_width = 174, 502, 10, 10
    image_patch = image[h_start:h_start + h_width, w_start:w_start + w_width]
    # Get maximum pixel values from each channel (BGR), normalize the original image
    # with these max pixel values - assuming the max pixel is white.
    image_normalized = image / image_patch.max(axis=(0, 1))
    # Some values will be above 1, so we need to clip the values to between 0 and 1
    image_balanced = image_normalized.clip(0, 1)    # to display
    cv2.rectangle(image, (w_start, h_start), (w_start + w_width, h_start + h_width), (0, 0, 255), 2)
    return image_balanced


def mouse_click_event(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = clone[y, x, 0]
        g = clone[y, x, 1]
        r = clone[y, x, 2]
        font_color = int(abs(255 - r)), int(abs(255 - g)), int(abs(255 - b))
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(clone, f'{x=}, {y=}', (x, y), font, 2, font_color, 2)
        cv2.imshow(f"{path} - pick the white spot", clone)
    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.putText(clone, f'{r=}, {g=}, {b=}', (x, y), font, 2, font_color, 2)
        cv2.imshow(f"{path} - pick the white spot", clone)


def show_image(image, window_name, with_mouse_callback=False):
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, open_photo_type)
    cv2.imshow(window_name, image)
    if with_mouse_callback:
        cv2.setMouseCallback(window_name, mouse_click_event)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


# reading the image; Flags=1: Any transparency of image will not be neglected
img = cv2.imread(path, flags=None if Method == "Gray-world algorithm" else 1)
if Method == "White patch reference":
    clone = img.copy()
    show_image(img, f"{path} - pick the white spot", with_mouse_callback=True)

if Method == "Gray-world algorithm":
    img_edit = gray_world_white_balance(img)
else:
    clone = img.copy()
    img_edit = white_patch_reference(clone)

if open_photo_before_edit:
    show_image(img, f"{path} - before edit")
if open_photo_after_edit:
    show_image(img_edit, f"{path} - after edit")
if Method == "White patch reference":
    # Convert to 8 bit before saving
    img_edit = (img_edit * 255).astype(int)
cv2.imwrite(" (edited).".join(path.rsplit(".", 1)), img_edit)
