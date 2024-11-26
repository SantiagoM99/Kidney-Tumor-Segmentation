"""
This module provides utility functions for analyzing, visualizing, and summarizing 3D medical imaging data. 
It includes the following functionalities:

1. Counting the orientation types of 3D medical images in a dataset.
2. Calculating slice statistics (minimum, maximum, average) for 3D scans.
3. Visualizing a slice from a 3D medical image with a corresponding segmentation mask.

Each function is designed for modular use and can be integrated into larger pipelines for medical image analysis.
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.orientations import aff2axcodes


def count_orientations(data_dir):
    """
    Counts the orientation types of 3D medical images in a dataset.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing case folders with NIfTI files.

    Returns
    -------
    dict
        A dictionary with orientation codes (e.g., "RAS", "LAS") as keys 
        and their counts as values.

    Notes
    -----
    Each case folder should contain a file named `filtered_imaging.nii.gz`.
    """
    orientation_counts = {}

    for case_folder in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        image_path = os.path.join(case_path, "filtered_imaging.nii.gz")
        if not os.path.exists(image_path):
            print(f"Image file not found for {case_folder}")
            continue

        # Load image and determine orientation
        image_nifti = nib.load(image_path)
        affine = image_nifti.affine
        orientation = ''.join(aff2axcodes(affine))

        # Update orientation counts
        if orientation not in orientation_counts:
            orientation_counts[orientation] = 0
        orientation_counts[orientation] += 1

    return orientation_counts


def compute_slice_statistics(data_dir):
    """
    Computes slice statistics (minimum, maximum, average) for 3D medical scans.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing case folders with NIfTI files.

    Returns
    -------
    dict
        A dictionary containing:
        - "total_scans": Total number of scans.
        - "min_slices": Minimum number of slices.
        - "max_slices": Maximum number of slices.
        - "avg_slices": Average number of slices.

    Notes
    -----
    Each case folder should contain a file named `filtered_imaging.nii.gz`.
    """
    slice_counts = []

    for case_folder in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        image_path = os.path.join(case_path, "filtered_imaging.nii.gz")
        if not os.path.exists(image_path):
            print(f"Image file not found for {case_folder}")
            continue

        # Load image and count slices
        image_nifti = nib.load(image_path)
        num_slices = image_nifti.shape[0]
        slice_counts.append(num_slices)

    # Compute statistics
    return {
        "total_scans": len(slice_counts),
        "min_slices": min(slice_counts),
        "max_slices": max(slice_counts),
        "avg_slices": sum(slice_counts) / len(slice_counts)
    }


def visualize_slice(image_path, mask_path, slice_idx=None):
    """
    Visualizes a specific slice from a 3D medical scan and its segmentation mask.

    Parameters
    ----------
    image_path : str
        Path to the NIfTI file for the 3D image volume.
    mask_path : str
        Path to the NIfTI file for the corresponding segmentation mask.
    slice_idx : int, optional
        Index of the slice to visualize. If None, the middle slice is used.

    Returns
    -------
    None
        Displays the image slice and the segmentation mask overlay.

    Notes
    -----
    The image and mask should have the same dimensions.
    """
    # Load image and mask
    image_nifti = nib.load(image_path)
    mask_nifti = nib.load(mask_path)

    image = image_nifti.get_fdata()
    mask = mask_nifti.get_fdata()

    # Choose slice index
    if slice_idx is None:
        slice_idx = image.shape[0] // 2  # Default to middle slice

    image_slice = image[slice_idx, :, :]
    mask_slice = mask[slice_idx, :, :]

    # Plot the image and mask
    plt.figure(figsize=(12, 6))

    # Raw image slice
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"Image Slice {slice_idx}")
    plt.axis("off")

    # Mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap="gray")  # Base image
    plt.imshow(mask_slice, alpha=0.5, cmap="Reds")  # Overlay mask
    plt.title(f"Mask Overlay Slice {slice_idx}")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    # Example usage
    data_dir = "./filtered_data"

    # Count orientations
    orientations = count_orientations(data_dir)
    print("\nOrientation Counts:")
    for orientation, count in orientations.items():
        print(f"{orientation}: {count} scans")

    # Compute slice statistics
    stats = compute_slice_statistics(data_dir)
    print("\nSlice Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Visualize a slice
    image_path = "filtered_data/case_00000/filtered_imaging.nii.gz"
    mask_path = "filtered_data/case_00000/filtered_segmentation.nii.gz"
    visualize_slice(image_path, mask_path)
