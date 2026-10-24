"""Two-head lesion classifier: BI-RADS 4-5 proxy (binary) + finding BI-RADS class."""
import timm
import torch
from torch import nn

from mammo.config import BIRADS_CLASSES

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class LesionClassifier(nn.Module):
    def __init__(self, arch: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.3)
        self.malignant_head = nn.Linear(dim, 1)
        self.birads_head = nn.Linear(dim, len(BIRADS_CLASSES))

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.malignant_head(f).squeeze(1), self.birads_head(f)


def load_classifier(path, device="cpu") -> LesionClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    checkpoint_classes = ckpt.get("birads_classes")
    if checkpoint_classes != BIRADS_CLASSES:
        raise ValueError(
            f"incompatible BI-RADS classes in classifier checkpoint: "
            f"{checkpoint_classes!r}; expected {BIRADS_CLASSES!r}"
        )
    model = LesionClassifier(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
