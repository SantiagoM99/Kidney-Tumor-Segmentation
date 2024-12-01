import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm



def flatten_slices_with_segmentation(data_dir, output_dir):
    """
    Processes 3D medical imaging data to extract slices with segmentation values > 0,
    and saves each slice as a separate image-mask pair in JPEG format with unique IDs.
    Handles multiple classes in segmentation by assigning different intensities.
    """
    # Define output directories for flattened images and masks
    flattened_dir = os.path.join(output_dir, "flattened_data")
    images_dir = os.path.join(flattened_dir, "images")
    masks_dir = os.path.join(flattened_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    unique_id = 0

    # Iterate over all cases
    for case_folder in tqdm(os.listdir(data_dir), desc="Processing cases"):
        case_path = os.path.join(data_dir, case_folder)
        if not os.path.isdir(case_path):
            continue

        # Paths to imaging and segmentation files
        image_path = os.path.join(case_path, "filtered_imaging.nii.gz")
        segmentation_path = os.path.join(case_path, "filtered_segmentation.nii.gz")

        if not (os.path.exists(image_path) and os.path.exists(segmentation_path)):
            continue

        # Load the image and segmentation files
        image_nifti = nib.load(image_path)
        segmentation_nifti = nib.load(segmentation_path)

        image = image_nifti.get_fdata(dtype=np.float32)
        segmentation = segmentation_nifti.get_fdata(dtype=np.float32)

        # Round the segmentation values to integers
        segmentation = np.round(segmentation).astype(np.uint8)

        # Process each slice in the 3D volume
        for slice_idx in range(image.shape[0]):
            segmentation_slice = segmentation[slice_idx, :, :]

            # Skip slices with no meaningful segmentation data
            if np.count_nonzero(segmentation_slice) == 0:
                continue

            # Normalize image slice for JPEG format (0-255)
            image_slice = image[slice_idx, :, :]
            if np.max(image_slice) == np.min(image_slice):  # Skip uniform slices
                continue

            image_slice_normalized = ((image_slice - np.min(image_slice)) / 
                                      (np.max(image_slice) - np.min(image_slice)) * 255).astype(np.uint8)

            
            # Skip slices without any class 1 or 2 segmentation
            if np.count_nonzero(segmentation_slice) == 0:
                continue

            # Create unique file IDs
            image_filename = os.path.join(images_dir, f"image_{unique_id:06d}.jpg")
            mask_filename = os.path.join(masks_dir, f"mask_{unique_id:06d}.jpg")

            # Save slices as npy files
            np.save(image_filename.replace(".jpg", ".npy"), image_slice_normalized)
            np.save(mask_filename.replace(".jpg", ".npy"), segmentation_slice)

            unique_id += 1

    print(f"Flattened dataset created at {flattened_dir}")



# Example usage
if __name__ == "__main__":
    data_dir = "./filtered_data"  # Input directory containing case folders
    output_dir = ""  # Output directory for flattened dataset
    flatten_slices_with_segmentation(data_dir, output_dir)
