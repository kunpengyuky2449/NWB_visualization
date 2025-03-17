# NWBenv
 
# Neural Data Analysis & Visualization

This repository contains tools and tutorials for **neural data analysis**, including **NWB file handling, dPCA for neural data dimensionality reduction, and a graphical UI for neuron figure inspection**.

## 📂 Repository Contents

### 1️⃣ **Tutorials**
- **`NWB_tutorial.ipynb`**  
  - A Jupyter Notebook tutorial on loading and processing **neural data in NWB format**.
  - Covers extracting spike times, trial events, unit metrices and basic analysis/visualization workflows.
  - Provide several utilities for visualzing unit quaities .

- **`dPCA_tutorial_playground.ipynb`**  
  - A simple starting hands-on notebook for performing **demixed Principal Component Analysis (dPCA)** on NWB file processed in previous pipeline.
  - Provide several utilities to construct input multidimensional tensor for dPCA from NWB file.

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
   pip install -r requirements.txt

