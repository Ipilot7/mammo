"""Train the image-level BI-RADS model (breast category 1-5) on whole VinDr-Mammo images."""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from mammo.config import DATA, LABELS_CSV, MODELS
from mammo.preprocess import to_gray_u8
from mammo.screening import (
    BREAST_BIRADS,
    ScreeningModel,
    screening_image,
    suspicious_prob,
    to_tensor,
)
from train.train_classifier import (
    autocast,
    pick_device,
    scaled_optimizer_step,
    to_device,
)


def image_table(split: str | None = None) -> pd.DataFrame:
    """One row per image with its breast BI-RADS (1-5)."""
    df = pd.read_csv(LABELS_CSV)
    if split:
        df = df[df.split == split]
    img = df.groupby("image_path", sort=False).agg(breast_birads=("breast_birads", "first")).reset_index()
    img = img[img.breast_birads.isin(BREAST_BIRADS)]
    img["label"] = img.breast_birads.astype(int) - 1
    return img[[(DATA / p).exists() for p in img.image_path]].reset_index(drop=True)


class ScreeningDataset(Dataset):
    def __init__(self, df, train: bool):
        self.df = df
        self.aug = T.Compose([T.RandomAffine(10, translate=(0.03, 0.03), scale=(0.9, 1.1)),
                              T.ColorJitter(0.25, 0.25)]) if train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = screening_image(to_gray_u8(Image.open(DATA / r.image_path)))
        if self.aug:
            img = self.aug(img)
        return to_tensor(img), torch.tensor(int(r.label))


def screening_metrics(probs, birads, threshold=None, target_sensitivity=0.9) -> dict:
    """probs: (N, 5) class probabilities; birads: true breast BI-RADS 1-5.

    - auc_1_2_vs_3_5: healthy/benign vs needs attention, scored by P(BI-RADS >= 3);
    - auc_1_2_vs_4_5: BI-RADS 1-2 vs 4-5, scored by P(4) + P(5);
    - threshold on P(>= 3) for target_sensitivity, with sensitivity/specificity.
    """
    probs, birads = np.asarray(probs), np.asarray(birads)
    p_susp, p_birads_4_5 = suspicious_prob(probs), probs[:, 3:].sum(1)
    sick = birads >= 3
    out = {"birads_acc": float((probs.argmax(1) + 1 == birads).mean())}
    if len(set(sick)) == 2:
        out["auc_1_2_vs_3_5"] = float(roc_auc_score(sick, p_susp))
    keep = (birads <= 2) | (birads >= 4)
    if len(set(birads[keep] >= 4)) == 2:
        out["auc_1_2_vs_4_5"] = float(roc_auc_score(birads[keep] >= 4, p_birads_4_5[keep]))
    if threshold is None and sick.any():
        threshold = float(np.quantile(p_susp[sick], 1 - target_sensitivity))
    if threshold is not None:
        pred = p_susp >= threshold
        out["threshold"] = threshold
        out["sensitivity"] = float(pred[sick].mean()) if sick.any() else float("nan")
        out["specificity"] = float((~pred[~sick]).mean()) if (~sick).any() else float("nan")
    out["confusion"] = confusion_matrix(birads, probs.argmax(1) + 1, labels=BREAST_BIRADS).tolist()
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        # Full precision is intentional: rare mammograms can overflow B3 activations
        # during FP16 validation even when the trained parameters remain finite.
        logits = model(to_device(x, device).float())
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(probs).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="efficientnet_b3")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--samples-per-epoch", type=int, default=6000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    device = pick_device(args.device)

    train_df, val_df = image_table("train"), image_table("val")
    if args.limit:
        train_df = train_df.groupby("label").head(args.limit).reset_index(drop=True)
        val_df = val_df.groupby("label").head(args.limit).reset_index(drop=True)
    print("train BI-RADS:", train_df.breast_birads.value_counts().sort_index().to_dict())
    print("val BI-RADS:", val_df.breast_birads.value_counts().sort_index().to_dict(), "device:", device)

    # Square-root balancing: rare BI-RADS 4-5 are seen far more often than their share,
    # but BI-RADS 1-2 still dominate, so probabilities on healthy breasts stay sensible.
    counts = train_df.groupby("label").image_path.transform("size").values
    n = min(args.samples_per_epoch, len(train_df)) if args.limit else args.samples_per_epoch
    sampler = WeightedRandomSampler(torch.tensor(counts ** -0.5, dtype=torch.double), n)
    train_dl = DataLoader(ScreeningDataset(train_df, True), args.batch, sampler=sampler, num_workers=args.workers,
                          drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(ScreeningDataset(val_df, False), args.batch, num_workers=args.workers)

    model = ScreeningModel(args.arch).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = -1.0
    MODELS.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = to_device(x, device), y.to(device)
            with autocast(device):
                logits = model(x)
            loss = ce(logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaled_optimizer_step(scaler, opt, sched)
            total += loss.item()
        m = screening_metrics(predict(model, val_dl, device), val_df.breast_birads)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train_dl):.4f} "
              f"{ {k: v for k, v in m.items() if k != 'confusion'} }", flush=True)
        score = m.get("auc_1_2_vs_3_5", m["birads_acc"])
        if score > best:
            best = score
            torch.save(
                {
                    "arch": args.arch,
                    "birads_classes": BREAST_BIRADS,
                    "state_dict": model.state_dict(),
                    "threshold": m.get("threshold", 0.5),
                    "val": m,
                },
                MODELS / "screening.pt",
            )
            print(f"  saved {MODELS / 'screening.pt'} (threshold {m.get('threshold', 0.5):.3f}); "
                  f"confusion rows=true BI-RADS 1-5: {m['confusion']}", flush=True)


if __name__ == "__main__":
    main()
