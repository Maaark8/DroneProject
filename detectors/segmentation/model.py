from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised through runtime guard
    torch = None
    nn = None


if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.layers(x)


    class SimpleUNet(nn.Module):
        def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
            super().__init__()
            self.enc1 = ConvBlock(in_channels, 16)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = ConvBlock(16, 32)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(32, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(32, 16)
            self.head = nn.Conv2d(16, out_channels, kernel_size=1)

        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            bottleneck = self.bottleneck(self.pool2(enc2))
            dec2 = self.up2(bottleneck)
            dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
            dec1 = self.up1(dec2)
            dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
            return self.head(dec1)
else:
    class SimpleUNet:  # pragma: no cover - exercised through runtime guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for the segmentation model.")
