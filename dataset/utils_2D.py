import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split
from Kits2019_2D import Kits20192DDataset
import os


def prepare_datasets(config, train_transform_type="train"):
    """
    Prepare datasets for training, validation, and testing based on the provided configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing paths, model settings, and split ratios. The dictionary should have the following keys:
        - "image_paths" (list of str): List of paths to all images.
        - "mask_paths" (list of str): List of paths to all corresponding masks.
        - "split_train" (float): Proportion of data to use for training.
        - "split_val" (float): Proportion of data to use for validation.
        - "split_test" (float): Proportion of data to use for testing.
        - "image_size" (int): Size to which images should be resized for training and testing.
    train_transform_type : str, optional
        Specifies the type of transformation to use for training (default is "train").

    Returns
    -------
    tuple
        A tuple containing three datasets:
        - train_dataset (Kits20192DDataset): The dataset for training.
        - val_dataset (Kits20192DDataset): The dataset for validation.
        - test_dataset (Kits20192DDataset): The dataset for testing.
    """
    # Unpack the configuration
    image_paths = config["image_paths"]
    mask_paths = config["mask_paths"]
    split_train = config["split_train"]
    split_val = config["split_val"]
    split_test = config["split_test"]
    image_size = config["image_size"]

    # Split into training, validation, and testing sets
    img_train, img_temp, mask_train, mask_temp = train_test_split(
        image_paths, mask_paths, test_size=(split_val + split_test), random_state=42
    )
    val_ratio = split_val / (split_val + split_test)
    img_val, img_test, mask_val, mask_test = train_test_split(
        img_temp, mask_temp, test_size=1 - val_ratio, random_state=42
    )

    # Create datasets
    train_dataset = Kits20192DDataset(
        img_train, mask_train, transform=get_transforms(train_transform_type, image_size)
    )
    val_dataset = Kits20192DDataset(
        img_val, mask_val, transform=get_transforms("test", image_size)
    )
    test_dataset = Kits20192DDataset(
        img_test, mask_test, transform=get_transforms("test", image_size)
    )

    return train_dataset, val_dataset, test_dataset


def get_transforms(phase, image_size):
    """
    Get the appropriate transforms for the specified phase.

    Parameters
    ----------
    phase : str
        The phase of the transformation. Must be one of 'train' or 'test'.
    image_size : int
        The size to which the images will be resized (height and width will be equal).

    Returns
    -------
    A.Compose
        A composition of transformations from the Albumentations library.

    Raises
    ------
    ValueError
        If an unsupported phase is specified.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
                    ],
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    elif phase == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Phase {phase} is not supported. Supported phases are 'train' and 'test'."
        )
    
def build_dataset_paths(images_dir, masks_dir, image_ext=".jpg", mask_ext=".jpg"):
    """
    Build lists of image and mask file paths by scanning the given directories.

    This function scans the provided directories for image and mask files, matches them
    by filename, and returns sorted lists of paths to ensure consistent pairing.

    Parameters
    ----------
    images_dir : str
        Path to the directory containing image files.
    masks_dir : str
        Path to the directory containing mask files.
    image_ext : str, optional
        File extension of the image files (default is ".jpg").
    mask_ext : str, optional
        File extension of the mask files (default is ".jpg").

    Returns
    -------
    tuple
        A tuple containing:
        - image_paths (list of str): Sorted list of image file paths.
        - mask_paths (list of str): Sorted list of corresponding mask file paths.

    Raises
    ------
    ValueError
        If there are unmatched images or masks.
    """
    # Get all image and mask files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(image_ext)])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(mask_ext)])

    # Ensure consistent pairing of image and mask files
    image_paths = []
    mask_paths = []
    for img_file in image_files:
        mask_file = img_file.replace(image_ext, mask_ext).replace("image", "mask")
        
        if mask_file in mask_files:
            image_paths.append(os.path.join(images_dir, img_file))
            mask_paths.append(os.path.join(masks_dir, mask_file))
        else:
            raise ValueError(f"Mask file {mask_file} not found for image {img_file}")

    return image_paths, mask_paths