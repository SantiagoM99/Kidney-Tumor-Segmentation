import torch
from PIL import Image
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, Rotate
from albumentations.pytorch import ToTensorV2

class Kits20192DDataset:
    """
    A custom PyTorch Dataset for loading and processing 2D medical scans
    from the KiTS19 dataset. This dataset assumes that the paths to images
    and masks are explicitly provided as separate lists.

    Parameters
    ----------
    image_paths : list of str
        A list of file paths to the image files.
        Example: ["data/images/image_000001.jpg", "data/images/image_000002.jpg", ...]
    mask_paths : list of str
        A list of file paths to the corresponding mask files.
        Example: ["data/masks/mask_000001.jpg", "data/masks/mask_000002.jpg", ...]
    transform : callable, optional
        An Albumentations transformation pipeline to be applied to the images
        and masks. If `None`, no transformations will be applied.

    Attributes
    ----------
    image_paths : list of str
        List of file paths to the image files.
    mask_paths : list of str
        List of file paths to the mask files.
    transform : callable or None
        The Albumentations transformation pipeline applied to the dataset.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Retrieves the image and mask pair for a given index, applies
        transformations if provided, and returns the processed tensors.

    """

    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Initializes the dataset with lists of image and mask paths, and an optional transform.

        Parameters
        ----------
        image_paths : list of str
            List of file paths to the image files.
        mask_paths : list of str
            List of file paths to the corresponding mask files.
        transform : callable, optional
            An Albumentations transformation pipeline to apply to the dataset.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of cases in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads the image and mask for the given index, applies transformations,
        and returns the processed tensors.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        torch.Tensor
            The transformed image tensor with shape `[C, H, W]`.
        torch.Tensor
            The transformed mask tensor with shape `[H, W]`.

        Notes
        -----
        - The mask values are remapped from the original dataset format:
          - 0: Background
          - 127: Kidney -> 1
          - 255: Tumor -> 2
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = np.load(image_path)
        mask = np.load(mask_path)
        # Get paths for image and mask
        # Convert mask to binary segmentation:
        # 0: background
        # 1: kidney (values in the middle range)
        # 2: tumor (values in the high range)
        mask = mask.astype(np.float32)


        # Apply Albumentations transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = torch.unsqueeze(mask, 0)

        return image, mask
