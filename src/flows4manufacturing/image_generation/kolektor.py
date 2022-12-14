import os
from glob import glob
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KolektorDataset(Dataset):
    def __init__(self, dir: str):
        """
        Args:
            dir: Path to the folder containing the images.
        """

        all_files = glob(os.path.join(dir, "*.png"))
        self._image_files = [x for x in all_files if "_GT" not in x and "copy" not in x]
        mask_files = [x for x in all_files if "_GT" in x and "copy" not in x]
        self._image_files.sort()
        mask_files.sort()

        # Load the labels
        self._labels = []
        for i, mask in enumerate(mask_files):
            assert (
                self._image_files[i][:-4] in mask
            ), f"Mask {mask} unexpectedly aligned with {self._image_files[i]}"
            loaded_mask = np.array(Image.open(mask).convert("L"))
            if (loaded_mask == 255).any():
                self._labels.append(1)
            else:
                self._labels.append(0)

        self._transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # Dequantization
                transforms.Lambda(lambda x: x + torch.rand_like(x) / 256),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image = Image.open(self._image_files[idx]).convert("L")
        return self._transform(image), self._labels[idx]  # type: ignore
