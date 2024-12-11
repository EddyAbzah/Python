"""
pyinstaller --noconfirm --onedir --windowed --contents-directory "Keyboard Shortcuts GUI" --icon "Icon.png" "Keyboard Shortcuts GUI.py"
"""


import re
import os
import sys
import psutil
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def get_shortcuts_data():
    full_shortcuts = {}
    if getattr(sys, 'frozen', False):  # Check if it's a PyInstaller executable
        path = os.path.dirname(sys.executable)
    else:
        path = os.path.dirname(os.path.abspath(__file__))
    for file_name in os.listdir(path):
        if ".txt" in file_name:
            # print(f'{file_name = }')
            program_shortcuts = []
            file_path = os.path.join(path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    match = re.match(r"^(.*?)\t.*\t(.*)$", line)
                    if match:
                        program_shortcuts.append((match.group(1), match.group(2)))
            full_shortcuts[os.path.splitext(os.path.basename(file_name))[0]] = program_shortcuts
    if not full_shortcuts:
        messagebox.showwarning("No Data Found", f"No keyboard shortcuts data found in the directory:\n{path}")
        sys.exit()
    return full_shortcuts


def get_current_program():
    running_processes = [proc.info["name"] for proc in psutil.process_iter(["name"])]
    running_processes = [string[:5].lower() for string in running_processes]
    my_shortcuts = [string[:5].lower() for string in shortcuts_data.keys()]
    matching_program_index = [index for index, program in enumerate(my_shortcuts) if program.lower() in running_processes]
    if matching_program_index:
        return list(shortcuts_data.keys())[matching_program_index[0]]
    else:
        # print("No matching programs found.")
        if "Windows" in shortcuts_data.keys():
            return "Windows"
        else:
            return list(shortcuts_data.keys())[0]


def set_window_properties():
    width = 400
    height = 50     # initial height for window elements
    height += len(shortcuts_data[current_program]) * 20
    root.geometry(f"{width}x{height}")
    root.title(f"Keyboard Shortcuts - {current_program}")


def toggle_always_on_top():
    global always_on_top
    always_on_top = not always_on_top
    root.attributes("-topmost", always_on_top)
    context_menu.entryconfig("Always on Top", variable=always_on_top_var)


def switch_program(program):
    global current_program
    current_program = program
    update_table()
    set_window_properties()
    for label, var in program_vars.items():
        var.set(label == program)


def update_table():
    tree.delete(*tree.get_children())  # Clear existing rows
    for index, (shortcut, action) in enumerate(shortcuts_data[current_program]):
        row_tag = 'odd' if index % 2 == 0 else 'even'  # Alternate tags for banded effect
        tree.insert("", tk.END, values=(shortcut, action), tags=(row_tag,))


def show_context_menu(event):
    context_menu.post(event.x_root, event.y_root)


shortcuts_data = get_shortcuts_data()
current_program = get_current_program()

root = tk.Tk()
set_window_properties()
always_on_top = False

# Create a frame for the table
frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.BOTH, expand=True)

# Create a Treeview widget to display the table
columns = ("Action", "Shortcut")
tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)

# Define columns
tree.heading("Action", text="Action")
tree.heading("Shortcut", text="Shortcut")
tree.column("Action", width=180, anchor="w")
tree.column("Shortcut", width=180, anchor="center")

# Add a scrollbar
scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Pack the Treeview
tree.pack(fill=tk.BOTH, expand=True)

# Configure banded row colors
tree.tag_configure('odd', background='#f5f5f5')  # Light gray for odd rows
tree.tag_configure('even', background='#ffffff')  # White for even rows

# Create a context menu
context_menu = tk.Menu(root, tearoff=0)
always_on_top_var = tk.BooleanVar(value=always_on_top)
context_menu.add_checkbutton(label="Always on Top", command=toggle_always_on_top, variable=always_on_top_var)
context_menu.add_separator()

# Add program switching options directly in the main menu with checkmarks
program_vars = {}
context_menu_labels = list(shortcuts_data.keys())
for program in shortcuts_data:
    program_vars[program] = tk.BooleanVar(value=(program == current_program))
    context_menu.add_checkbutton(label=program, command=lambda p=program: switch_program(p), variable=program_vars[program])

tree.bind("<Button-3>", show_context_menu)
update_table()
root.mainloop()
