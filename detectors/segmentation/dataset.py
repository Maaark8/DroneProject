from __future__ import annotations

from pathlib import Path

import cv2

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised through runtime guard
    torch = None
    Dataset = object


class TrackSegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path) -> None:
        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        self.items: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            mask_path = masks_dir / image_path.name
            if mask_path.exists():
                self.items.append((image_path, mask_path))

        if not self.items:
            raise ValueError("No matching image/mask pairs found for segmentation training.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, mask_path = self.items[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)
        return image_tensor, mask_tensor
