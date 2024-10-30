# pyinstaller --noconfirm --onefile --windowed --contents-directory "Random Ads" --icon "Random Ads.ico"  "Random Ads.py"


import tkinter as tk
import random
import time
import threading
import queue

# Timer for pop-up interval in seconds
timer = 60

# List of ad-like messages
messages = [
    "Congratulations! You've won a free trip!",
    "Exclusive offer just for you: Click here!",
    "Hot singles in your area are waiting for you!",
    "Your computer is infected! Click to fix!",
    "You've been selected for a special promotion!",
    "Act now to claim your free gift!",
    "Limited time offer! Don't miss out!",
    "You've been chosen for a secret survey!",
    "Your account has been compromised! Verify now!",
    "You are the lucky visitor of the day!",
    "Claim your prize before it’s too late!",
    "You have a new message from an admirer!",
    "Your download is ready! Click here!",
    "You could be the next big winner!",
    "Urgent: Update your software now!",
]


def random_color():
    """Generate a random color in hex format."""
    return f'#{random.randint(0, 0xFFFFFF):06x}'


def create_popup():
    # Randomly select a message
    message = random.choice(messages)

    # Create a new borderless Tkinter window
    popup = tk.Toplevel()
    popup.overrideredirect(True)  # Remove title bar and borders
    popup.geometry("600x300")  # Set a size for the popup (twice as big)

    # Get screen width and height
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    # Generate random x and y coordinates
    x = random.randint(0, screen_width - 600)  # Width of popup is 600
    y = random.randint(0, screen_height - 300)  # Height of popup is 300

    # Set the position of the popup
    popup.geometry(f"+{x}+{y}")

    # Random colors for text and button
    text_color = random_color()
    button_color = random_color()

    # Create a label with a much larger font size
    label_message = tk.Label(popup, text=message, padx=20, pady=50, fg=text_color, font=("Helvetica", 20))
    label_message.pack()

    # Close button with random color
    button = tk.Button(popup, text="Claim Now", command=popup.destroy, bg=button_color, font=("Helvetica", 20))
    button.pack(pady=10)

    # Set focus to the popup
    popup.lift()  # Bring the window to the front


def popup_generator():
    while True:
        # Add a task to the queue to create a pop-up
        queue.put(create_popup)
        time.sleep(timer)  # Sleep for the specified timer interval


def check_queue():
    while not queue.empty():
        func = queue.get()
        func()  # Call the function from the queue
    root.after(100, check_queue)  # Check the queue again after 100 ms


# Create a queue for communication between threads
queue = queue.Queue()

# Create the main Tkinter window but immediately hide it
root = tk.Tk()
root.withdraw()  # Hide the main window

# Start the generator in a separate thread
thread = threading.Thread(target=popup_generator)
thread.daemon = True
thread.start()

# Start checking the queue
root.after(100, check_queue)
tk.mainloop()
