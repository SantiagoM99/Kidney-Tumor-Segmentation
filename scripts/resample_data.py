"""
This script filters 3D medical imaging scans to retain only the slices that have non-zero
values in the segmentation mask. It processes all cases in a dataset, saves the filtered
images and masks into an output directory, and ensures the data integrity by preserving
the affine matrix and header information.

Inputs:
- Each case folder in `data_dir` must contain:
  - `imaging.nii.gz`: The 3D image volume.
  - `segmentation.nii.gz`: The corresponding 3D segmentation mask.

Outputs:
- Filtered files for each case are saved in `output_dir`:
  - `filtered_imaging.nii.gz`: Filtered image slices.
  - `filtered_segmentation.nii.gz`: Filtered segmentation slices.

Usage:
- Set the `data_dir` and `output_dir` paths.
- Call the `filter_slices_with_segmentation(data_dir, output_dir)` function.
"""

import os
import nibabel as nib
import numpy as np

def filter_slices_with_segmentation(data_dir, output_dir, target_slices=30, tolerance=0.5):
    """
    Filters 3D medical imaging data to retain only 30 slices, prioritizing slices
    with both kidney and tumor regions (using approximate segmentation values).
    Adds background slices if necessary.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing case folders with `imaging.nii.gz` and `segmentation.nii.gz`.
    output_dir : str
        Path to the directory where filtered files will be saved.
    target_slices : int
        The number of slices to retain for each case.
    tolerance : float
        The tolerance range for identifying segmentation values for kidney and tumor.

    Returns
    -------
    None
        The function processes all cases in the dataset and saves the filtered results
        in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for case_folder in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        # Paths to imaging and segmentation files
        image_path = os.path.join(case_path, "filtered_imaging.nii.gz")
        segmentation_path = os.path.join(case_path, "filtered_segmentation.nii.gz")

        if not (os.path.exists(image_path) and os.path.exists(segmentation_path)):
            print(f"Skipping {case_folder}: Missing imaging or segmentation file.")
            continue

        # Load the image and segmentation files
        image_nifti = nib.load(image_path)
        segmentation_nifti = nib.load(segmentation_path)

        image = image_nifti.get_fdata(dtype=np.float32)
        segmentation = segmentation_nifti.get_fdata(dtype=np.float32)

        # Define approximate ranges for kidney and tumor
        kidney_range = (1 - tolerance, 1 + tolerance)  # Approximate range for kidney
        tumor_range = (2 - tolerance, 2 + tolerance)   # Approximate range for tumor

        # Identify slices with kidney and tumor based on ranges
        kidney_slices = np.any((segmentation >= kidney_range[0]) & (segmentation <= kidney_range[1]), axis=(1, 2))
        tumor_slices = np.any((segmentation >= tumor_range[0]) & (segmentation <= tumor_range[1]), axis=(1, 2))
        both_present = kidney_slices & tumor_slices

        relevant_slices = np.where(both_present)[0].tolist()

        # Add slices with only kidney or tumor if needed
        if len(relevant_slices) < target_slices:
            additional_slices = np.where(kidney_slices | tumor_slices)[0].tolist()
            for slice_idx in additional_slices:
                if slice_idx not in relevant_slices:
                    relevant_slices.append(slice_idx)
                    if len(relevant_slices) >= target_slices:
                        break

        # If still less than target_slices, add background slices
        if len(relevant_slices) < target_slices:
            all_slices = list(range(image.shape[0]))
            for slice_idx in all_slices:
                if slice_idx not in relevant_slices:
                    relevant_slices.append(slice_idx)
                    if len(relevant_slices) >= target_slices:
                        break

        # Select exactly `target_slices` and maintain original spatial order
        relevant_slices = sorted(relevant_slices[:target_slices])

        # Filter the image and segmentation volumes
        filtered_image = image[relevant_slices, :, :]
        filtered_segmentation = segmentation[relevant_slices, :, :]

        # Save the filtered data as new NIfTI files
        filtered_image_nifti = nib.Nifti1Image(filtered_image, affine=image_nifti.affine, header=image_nifti.header)
        filtered_segmentation_nifti = nib.Nifti1Image(filtered_segmentation, affine=segmentation_nifti.affine, header=segmentation_nifti.header)

        # Create the output folder for the case
        output_case_dir = os.path.join(output_dir, case_folder)
        os.makedirs(output_case_dir, exist_ok=True)

        nib.save(filtered_image_nifti, os.path.join(output_case_dir, "filtered_imaging.nii.gz"))
        nib.save(filtered_segmentation_nifti, os.path.join(output_case_dir, "filtered_segmentation.nii.gz"))

        print(f"Processed and saved filtered data for {case_folder}")

# Example usage
if __name__ == "__main__":
    data_dir = "./filtered_data"  # Input directory containing case folders
    output_dir = "./data"  # Output directory for filtered cases
    filter_slices_with_segmentation(data_dir, output_dir)
