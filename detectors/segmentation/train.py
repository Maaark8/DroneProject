from __future__ import annotations

from pathlib import Path

from track_detection.io import ensure_directory

from .dataset import TrackSegmentationDataset, torch
from .model import SimpleUNet


def train_segmentation_model(
    images_dir: Path,
    masks_dir: Path,
    checkpoint_path: Path,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> None:
    if torch is None:
        raise ImportError("PyTorch is required for segmentation training. Install the 'train' extra.")

    from torch import nn
    from torch.utils.data import DataLoader

    dataset = TrackSegmentationDataset(images_dir=images_dir, masks_dir=masks_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for images, masks in loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

    ensure_directory(checkpoint_path.parent)
    torch.save(model.state_dict(), checkpoint_path)
