"""Image-level model: BI-RADS category (1-5) of the whole breast on a mammogram.

p_suspicious = P(BI-RADS 3) + P(4) + P(5) drives the "needs attention" verdict.
"""
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn

from mammo.classifier import MEAN, STD
from mammo.preprocess import crop_breast

INPUT_W, INPUT_H = 512, 1024
BREAST_BIRADS = [1, 2, 3, 4, 5]


def screening_image(gray: np.ndarray) -> Image.Image:
    """Breast crop, oriented with the chest wall on the left, letterboxed to INPUT_W x INPUT_H."""
    img, _ = crop_breast(gray, height=INPUT_H)
    a = np.asarray(img)
    half = a.shape[1] // 2
    if a[:, :half].mean() < a[:, half:].mean():  # chest wall = brighter side
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if img.width > INPUT_W:
        img = img.resize((INPUT_W, round(img.height * INPUT_W / img.width)), Image.BILINEAR)
    canvas = Image.new("L", (INPUT_W, INPUT_H))
    canvas.paste(img, (0, 0))
    return canvas


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    a = a.unsqueeze(0).repeat(3, 1, 1)
    return (a - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class ScreeningModel(nn.Module):
    def __init__(self, arch: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=len(BREAST_BIRADS),
                                          drop_rate=0.3)

    def forward(self, x):
        return self.backbone(x)


def suspicious_prob(probs):
    """P(BI-RADS >= 3) from class probabilities (..., 5)."""
    return probs[..., 2:].sum(-1)


def load_screening(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    checkpoint_classes = ckpt.get("birads_classes")
    if checkpoint_classes != BREAST_BIRADS:
        raise ValueError(
            f"incompatible BI-RADS classes in screening checkpoint: "
            f"{checkpoint_classes!r}; expected {BREAST_BIRADS!r}"
        )
    model = ScreeningModel(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), float(ckpt["threshold"])
