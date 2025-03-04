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


for folder in [left_folder, right_folder]:
    os.makedirs(folder, exist_ok=True)
images = [f for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
index = 0


def show_image():
    """Display the current image in the tkinter window or close if done."""
    if index >= len(images):
        root.destroy()  # Close app when all images are sorted
        return

    img_path = os.path.join(source_folder, images[index])
    image = Image.open(img_path)
    image.thumbnail((800, 600))  # Resize for display
    img_tk = ImageTk.PhotoImage(image)

    lbl.config(image=img_tk, text="")
    lbl.image = img_tk  # Keep reference


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


root = tk.Tk()
root.title("Image Sorter")
lbl = Label(root)
lbl.pack()

# Bind arrow keys to actions
root.bind("<Left>", lambda _: move_image(left_folder))      # Left Arrow → Move to left_folder
root.bind("<Right>", lambda _: move_image(right_folder))    # Right Arrow → Move to right_folder
root.bind("<Down>", lambda _: delete_image())               # Down Arrow → Send to Recycle Bin
root.bind("<Up>", lambda _: skip_image())                   # Up Arrow → Skip to next image

show_image()  # Show first image
root.mainloop()
