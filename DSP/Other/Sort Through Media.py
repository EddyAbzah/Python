"""
Open all pictures or files in a folder, and send to designated folder via the arrow keys.
Down arrow to delete; up arrow to skip.
"""


import os
import cv2
import shutil
import send2trash
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk


source_folder = r""
left_folder = r""
right_folder = r""


T_files__F_files = False
index = 0
current_file_path = None  # Track current image path
cap = None  # OpenCV video capture object
pending_action = None  # Store action to perform after skipping


if T_files__F_files:
    files = [f for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
else:
    files = [f for f in os.listdir(source_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
files.sort()


def show_image():
    """Display the current image in the Tkinter window or close if done."""
    global index, current_file_path
    if index >= len(files):
        root.destroy()  # Close app when all files are sorted
        return

    current_file_path = os.path.join(source_folder, files[index])
    resize_and_display_image()


def resize_and_display_image(event=None):
    """Resize the image to fit within the window while keeping its aspect ratio."""
    if not current_file_path:
        return

    # Get window size (ensuring it's valid)
    win_width = max(root.winfo_width(), 1)
    win_height = max(root.winfo_height(), 1)

    # Open image and get its original size
    image = Image.open(current_file_path)
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
    if index < len(files):
        src = os.path.join(source_folder, files[index])
        dest = os.path.join(destination, files[index])
        shutil.move(src, dest)
        index += 1
        show_image()  # Load next image


def delete_image():
    """Send the current image to the Recycle Bin and go to the next one."""
    global index
    if index < len(files):
        src = os.path.join(source_folder, files[index])
        send2trash.send2trash(src)  # Move file to Recycle Bin
        index += 1
        show_image()


def skip_image():
    """Skip the current image and go to the next one."""
    global index
    index += 1
    show_image()


def show_video():
    """Play the current video in the Tkinter window or close if done."""
    global index, current_file_path, cap, pending_action

    if index >= len(files):
        root.destroy()  # Close app when all files are sorted
        return

    # Process pending action before playing next video
    if pending_action:
        pending_action()
        pending_action = None

    # Load new video
    current_file_path = os.path.join(source_folder, files[index])
    cap = cv2.VideoCapture(current_file_path)
    play_video_frame()


def play_video_frame():
    """Read and display the next frame from the video."""
    global cap

    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()  # Video finished, go to next
        index_video(1)
        return

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get window size
    win_width = max(root.winfo_width(), 1)
    win_height = max(root.winfo_height(), 1)

    # Compute the new size while keeping aspect ratio
    img_height, img_width, _ = frame.shape
    scale = min(win_width / img_width, win_height / img_height)
    new_width = max(int(img_width * scale), 1)
    new_height = max(int(img_height * scale), 1)

    # Resize and convert to Tkinter format
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    img_tk = ImageTk.PhotoImage(Image.fromarray(resized_frame))

    lbl.config(image=img_tk)
    lbl.image = img_tk  # Keep reference to avoid garbage collection
    lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=new_width, height=new_height)  # Center image

    root.after(30, play_video_frame)  # Schedule next frame (approx. 30 FPS)


def index_video(step):
    """Move to the next or previous video."""
    global index, cap

    if cap:
        cap.release()
        cap = None

    index += step
    show_video()


def queue_action(action):
    """Skip the video and queue an action (move or delete)."""
    global pending_action
    pending_action = action
    index_video(1)  # Immediately skip to next video


def move_video(destination):
    """Move video to destination folder after skipping."""
    def action():
        src = os.path.join(source_folder, files[index - 1])
        dest = os.path.join(destination, files[index - 1])
        shutil.move(src, dest)
    queue_action(action)


def delete_video():
    """Send the current video to the Recycle Bin after skipping."""
    def action():
        src = os.path.join(source_folder, files[index - 1])
        send2trash.send2trash(src)  # Move file to Recycle Bin
    queue_action(action)


def skip_video():
    """Skip the current video and go to the next one immediately."""
    index_video(1)


if __name__ == '__main__':
    root = tk.Tk()
    if T_files__F_files:
        root.title("Image Sorter")
    else:
        root.title("Video Sorter")
    root.geometry("1000x800")  # Set initial window size
    root.minsize(400, 300)  # Prevent window from being too small

    lbl = Label(root)
    lbl.pack(fill=tk.BOTH, expand=True)  # Make label fill window

    # Bind arrow keys to actions
    if T_files__F_files:
        root.bind("<Left>", lambda _: move_image(left_folder))  # Left Arrow → Move to left_folder
        root.bind("<Right>", lambda _: move_image(right_folder))  # Right Arrow → Move to right_folder
        root.bind("<Down>", lambda _: delete_image())  # Down Arrow → Send to Recycle Bin
        root.bind("<Up>", lambda _: skip_image())  # Up Arrow → Skip to next image

        # Bind window resize event to dynamically scale the image
        root.bind("<Configure>", resize_and_display_image)
        # Delay first image display until the window is fully initialized
        root.after(100, show_image)
    else:
        root.bind("<Left>", lambda _: move_video(left_folder))  # Left Arrow → Move to left_folder
        root.bind("<Right>", lambda _: move_video(right_folder))  # Right Arrow → Move to right_folder
        root.bind("<Down>", lambda _: delete_video())  # Down Arrow → Send to Recycle Bin
        root.bind("<Up>", lambda _: skip_video())  # Up Arrow → Skip to next video

        # Bind window resize event to dynamically scale the video
        root.bind("<Configure>", lambda _: play_video_frame())
        # Delay first video display until the window is fully initialized
        root.after(100, show_video)

    root.mainloop()
