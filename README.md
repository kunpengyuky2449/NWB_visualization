# NWBenv
 
# Neural Data Analysis & Visualization

This repository contains tools and tutorials for **neural data analysis**, including **NWB file handling, dPCA for neural data dimensionality reduction, and a graphical UI for neuron figure inspection**.

## 📂 Repository Contents

### 1️⃣ **Tutorials**
- **`NWB_tutorial.ipynb`**  
  - A Jupyter Notebook tutorial on loading and processing **neural data in NWB format**.
  - Covers extracting spike times, trial events, and basic analysis workflows.

- **`dPCA_tutorial_playground.ipynb`**  
  - A hands-on notebook for performing **demixed Principal Component Analysis (dPCA)**.
  - Shows how to extract and visualize task-related neural population activity.

### 2️⃣ **Neuron Inspection UI**
- **`Neuron_inspection_UI.py`**  
  - A Tkinter-based **graphical user interface (GUI)** for **browsing, marking, and saving neuron figure selections**.
  - Key features:
    - Load neuron figure images from a folder.
    - Navigate using buttons or **keyboard shortcuts** (`←`, `→`, `Space`, `Ctrl+S`).
    - Mark/unmark neurons and save selections to a **JSON file**.

## 🛠 Installation & Dependencies

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
