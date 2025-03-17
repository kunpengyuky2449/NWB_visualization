"""
Neuron Figure Browser - UI for Reviewing and Selecting Neuron Figures

This application allows users to browse, mark, and save selections of neuron figures 
from a specified folder. The interface consists of an image viewer with navigation 
controls and a debug/error log panel.

Key Features:
- Load and display neuron figure images from a selected directory.
- Navigate through images using buttons or keyboard shortcuts (Left/Right arrows).
- Mark/unmark neurons and track selections.
- Save selected neurons to a JSON file.
- Debug/error log panel to track messages and errors.

Dependencies:
- tkinter (for UI elements)
- PIL (for image processing)
- json (for saving selections)
- pathlib (for handling file paths)
- traceback (for error logging)
- os (for file operations)

Global Variables:
- `image_list`: Stores the list of images loaded from the directory.
- `current_index`: Tracks the currently displayed image.
- `selected_neurons`: Stores marked neurons (electrode ID, unit ID).
- `session_name`: Extracted session identifier from filenames.
- `marked_neurons`: Tracks marked neuron filenames.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext
from pathlib import Path
import traceback
from PIL import Image, ImageTk

# Global variables
image_list = []
current_index = 0
selected_neurons = []
session_name = ""  # Stores session name for final save
marked_neurons = set()  # Stores marked neuron filenames for tracking

def log_message(message):
    """Logs messages to the debug/error panel in the UI."""
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)  # Auto-scroll to the latest message

def extract_neuron_info(filename):
    """Extracts session name, electrode name, and real unit number from a neuron figure filename."""
    filename = Path(filename).stem  # Remove file extension
    parts = filename.split("_")

    try:
        global session_name
        session_name = "_".join(parts[:2])  # Example: "exp2024-04-03-135625_prepro"

        electrode_part = next(p for p in parts if p.startswith("Electrode-"))
        electrode_name = electrode_part.replace("Electrode-", "")

        unit_part = next(p for p in parts if p.startswith("Unit-"))
        real_unit_number = int(unit_part.replace("Unit-", ""))

        return session_name, electrode_name, real_unit_number

    except (ValueError, StopIteration, IndexError) as e:
        log_message(f"❌ Error: Invalid filename format: {filename}\n{traceback.format_exc()}")
        return None, None, None

def update_selected_count():
    """Updates the label displaying the number of selected neurons."""
    selected_count_label.config(text=f"Selected: {len(selected_neurons)}")

def load_images():
    """Loads all images from a selected directory and resets session & selection state."""
    global image_list, current_index, session_name, selected_neurons, marked_neurons

    folder_selected = filedialog.askdirectory(title="Select Figure Folder")
    if not folder_selected:
        return

    # Reset session and selection
    selected_neurons.clear()
    marked_neurons.clear()
    session_name = ""

    image_list = [Path(folder_selected) / f for f in os.listdir(folder_selected) if f.endswith(".jpg")]
    image_list.sort()

    if image_list:
        current_index = 0
        session_name, _, _ = extract_neuron_info(image_list[0])
        log_message(f"📂 Loaded {len(image_list)} images from {folder_selected}")
        update_selected_count()  # Reset selection count display
        display_image()

def display_image():
    """Displays the current image."""
    global current_index, img_label, mark_label, selected_count_label

    if not image_list:
        return

    img_path = image_list[current_index]
    
    try:
        img = Image.open(img_path)
        img = img.resize((600, 600), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        img_label.config(image=img)
        img_label.image = img  # Keep reference

        mark_label.config(text="⭐" if img_path.name in marked_neurons else "")
        update_selected_count()  # Update selection count display

        title_label.config(text=f"Viewing: {img_path.name} ({current_index + 1}/{len(image_list)})")

    except Exception as e:
        log_message(f"❌ Error displaying image {img_path.name}\n{traceback.format_exc()}")

def prev_image():
    """Goes to the previous image."""
    global current_index
    if current_index > 0:
        current_index -= 1
        display_image()

def next_image():
    """Goes to the next image."""
    global current_index
    if current_index < len(image_list) - 1:
        current_index += 1
        display_image()

def mark_neuron():
    """Marks or unmarks the current neuron."""
    global selected_neurons, marked_neurons

    if not image_list:
        return

    img_path = image_list[current_index]
    session, electrode, unit = extract_neuron_info(img_path)

    if electrode and unit:
        neuron_info = (electrode, unit)

        if img_path.name in marked_neurons:
            marked_neurons.remove(img_path.name)
            selected_neurons.remove(neuron_info)
            status_label.config(text=f"Neuron Unmarked: Electrode-{electrode}, Unit-{unit}")
            log_message(f"❌ Unmarked neuron: {electrode}, Unit {unit}")
        else:
            marked_neurons.add(img_path.name)
            selected_neurons.append(neuron_info)
            status_label.config(text=f"Neuron Marked: Electrode-{electrode}, Unit-{unit}")
            log_message(f"✅ Marked neuron: {electrode}, Unit {unit}")

    display_image()  # Refresh image and count

def save_selection():
    """Saves the selected neurons along with the session name."""
    if not selected_neurons:
        status_label.config(text="⚠ No neurons selected!")
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save Selected Neurons"
    )

    if not save_path:
        return

    data_to_save = {
        "session_name": session_name,
        "selected_neurons": selected_neurons
    }

    try:
        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        status_label.config(text=f"✅ Selections saved to {save_path}")
        log_message(f"💾 Saved selections to {save_path}")

    except Exception as e:
        log_message(f"❌ Error saving selections\n{traceback.format_exc()}")

def handle_keypress(event):
    """Handles keyboard shortcuts."""
    if event.keysym == "Left":
        prev_image()
    elif event.keysym == "Right":
        next_image()
    elif event.keysym == "space":
        mark_neuron()
    elif event.state == 4 and event.keysym.lower() == "s":  # Ctrl + S
        save_selection()

# Initialize Tkinter window
root = tk.Tk()
root.title("Neuron Figure Browser")
root.geometry("1000x800")  # ✅ Increased default window size

# Bind keyboard events
root.bind("<Left>", handle_keypress)
root.bind("<Right>", handle_keypress)
root.bind("<space>", handle_keypress)
root.bind("<Control-s>", handle_keypress)

# Main layout: Left debug panel, right image & buttons
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Left panel for debug logs
debug_frame = tk.Frame(main_frame, width=300, height=800)
debug_frame.pack(side="left", fill="y", padx=5, pady=5)

log_label = tk.Label(debug_frame, text="🔍 Debug / Error Log", font=("Arial", 12))
log_label.pack()

log_text = scrolledtext.ScrolledText(debug_frame, wrap=tk.WORD, width=40, height=40)
log_text.pack(fill="both", expand=True)

# Right panel for image and controls
right_frame = tk.Frame(main_frame)
right_frame.pack(side="right", fill="both", expand=True)

title_label = tk.Label(right_frame, text="Neuron Figure Browser", font=("Arial", 14))
title_label.pack()

img_frame = tk.Frame(right_frame)
img_frame.pack()

img_label = tk.Label(img_frame)  # Image placeholder
img_label.pack(side="left")

mark_label = tk.Label(img_frame, text="", font=("Arial", 24), fg="gold")  # Star label
mark_label.pack(side="right", padx=10)

selected_count_label = tk.Label(right_frame, text="Selected: 0", font=("Arial", 12), fg="blue")
selected_count_label.pack()

button_frame = tk.Frame(right_frame)
button_frame.pack()

prev_button = tk.Button(button_frame, text="⬅ Prev", command=prev_image)
prev_button.grid(row=0, column=0, padx=5, pady=5)

mark_button = tk.Button(button_frame, text="✅ Mark Neuron", command=mark_neuron)
mark_button.grid(row=0, column=1, padx=5, pady=5)

next_button = tk.Button(button_frame, text="Next ➡", command=next_image)
next_button.grid(row=0, column=2, padx=5, pady=5)

save_button = tk.Button(right_frame, text="💾 Save Selection", command=save_selection)
save_button.pack(pady=5)

status_label = tk.Label(right_frame, text="", font=("Arial", 10), fg="green")
status_label.pack()

load_button = tk.Button(right_frame, text="📂 Load Figure Folder", command=load_images)
load_button.pack(pady=5)

root.mainloop()
