"""
Open all pictures in a folder, and send to designated folder via the arrow keys.
Down arrow to delete; up arrow to skip.
"""


import os
import shutil
import send2trash
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk


source_folder = r""
left_folder = r""
right_folder = r""


images = [f for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
images.sort()
index = 0
current_img_path = None  # Track current image path


def show_image():
    """Display the current image in the Tkinter window or close if done."""
    global index, current_img_path
    if index >= len(images):
        root.destroy()  # Close app when all images are sorted
        return

    current_img_path = os.path.join(source_folder, images[index])
    resize_and_display_image()


def resize_and_display_image(event=None):
    """Resize the image to fit within the window while keeping its aspect ratio."""
    if not current_img_path:
        return

    # Get window size (ensuring it's valid)
    win_width = max(root.winfo_width(), 1)
    win_height = max(root.winfo_height(), 1)

    # Open image and get its original size
    image = Image.open(current_img_path)
    img_width, img_height = image.size

    # Compute the new size while maintaining aspect ratio
    scale = min(win_width / img_width, win_height / img_height)
    new_width = max(int(img_width * scale), 1)
    new_height = max(int(img_height * scale), 1)

    # Resize and display the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(resized_image)

    lbl.config(image=img_tk)
    lbl.image = img_tk  # Keep reference to avoid garbage collection
    lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=new_width, height=new_height)  # Center image


def move_image(destination):
    """Move image to destination folder and go to next image."""
    global index
    if index < len(images):
        src = os.path.join(source_folder, images[index])
        dest = os.path.join(destination, images[index])
        shutil.move(src, dest)
        index += 1
        show_image()  # Load next image


def delete_image():
    """Send the current image to the Recycle Bin and go to the next one."""
    global index
    if index < len(images):
        src = os.path.join(source_folder, images[index])
        send2trash.send2trash(src)  # Move file to Recycle Bin
        index += 1
        show_image()


def skip_image():
    """Skip the current image and go to the next one."""
    global index
    index += 1
    show_image()


# MAIN EXECUTION BLOCK
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Sorter")
    root.geometry("800x600")  # Set initial window size
    root.minsize(400, 300)  # Prevent window from being too small

    lbl = Label(root)
    lbl.pack(fill=tk.BOTH, expand=True)  # Make label fill window

    # Bind arrow keys to actions
    root.bind("<Left>", lambda _: move_image(left_folder))   # Left Arrow → Move to left_folder
    root.bind("<Right>", lambda _: move_image(right_folder)) # Right Arrow → Move to right_folder
    root.bind("<Down>", lambda _: delete_image())           # Down Arrow → Send to Recycle Bin
    root.bind("<Up>", lambda _: skip_image())               # Up Arrow → Skip to next image

    # Bind window resize event to dynamically scale the image
    root.bind("<Configure>", resize_and_display_image)

    # Delay first image display until the window is fully initialized
    root.after(100, show_image)

    root.mainloop()
