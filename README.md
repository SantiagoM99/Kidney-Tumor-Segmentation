# Kidney-Tumor-Segmentation

## Project Overview

This repository is dedicated to kidney and renal tumor segmentation, developed as part of the **KiTS2019 Challenge**. The task involves correctly segmenting both kidneys and renal tumors from medical imaging datasets. The repository includes implementations for **3D segmentation** and **2D segmentation**, along with supporting scripts and tools for data preparation and experimentation.

---

## Features

- **3D Segmentation**: Implements volumetric analysis for more accurate segmentation of kidneys and tumors.
- **2D Segmentation**: A slice-based approach for scenarios where computational resources are limited.
- **Data Sampling Scripts**: Located in the `scripts` folder, these allow you to sample data in both 3D and 2D formats for testing and training.
- **Jupyter Notebooks**: Explore the models interactively. Notebooks are available in the `notebooks` folder and can be moved to the repository's root directory for easier usage.

---

## Setting Up the Environment

To work with this project, you need to create a Python virtual environment and install the required dependencies.

### Step 1: Create a Virtual Environment
Run the following commands in your terminal:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Step 2: Install Dependencies
With the virtual environment activated, install the required libraries:
```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

If you'd like to try the project interactively:
1. Copy a notebook of your choice from the `notebooks` folder to the root directory of the repository.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the notebook from the root directory and follow the instructions inside.

---

## Data Preparation Scripts

In the `scripts` folder, you will find utilities for data preparation:
- **3D Data Sampling**: Use the provided scripts to process volumetric data for training and testing.
- **2D Data Sampling**: Convert volumetric data into slices for 2D segmentation tasks.

Example usage:
```bash
python scripts/resample_data.py
python scripts/resample_data_2D.py
```

---

## Contribution and Exploration

This repository provides the building blocks for experimenting with kidney and renal tumor segmentation. Whether you're analyzing the performance of 3D versus 2D segmentation or preparing your dataset using the scripts, we encourage you to explore and contribute to this project.

Happy coding!