"""Train the BI-RADS 4-5 proxy / finding BI-RADS classifier on ground-truth lesion crops."""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from mammo.classifier import MEAN, STD, LesionClassifier
from mammo.config import BIRADS_CLASSES, DATA, LABELS_CSV, MODELS
from mammo.crops import crop_lesion


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train
        aug = [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.3, 0.3)] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA / r.image_path).convert("RGB")
        crop = crop_lesion(img, (r.x1, r.y1, r.x2, r.y2), jitter=0.25 if self.train else 0.0)
        mal = -1 if pd.isna(r.malignant) else int(r.malignant)
        bir = -1 if pd.isna(r.birads_idx) else int(r.birads_idx)
        return self.tf(crop), torch.tensor(mal, dtype=torch.float32), torch.tensor(bir)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    """fp16 mixed precision on CUDA (fits EfficientNet-B3 @384 into a T4); full precision elsewhere."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


def to_device(x, device):
    if device.type == "cuda":
        return x.to(device, memory_format=torch.channels_last, non_blocking=True)
    return x.to(device)


def scaled_optimizer_step(scaler, optimizer, scheduler) -> bool:
    """Advance the LR schedule only when GradScaler did not skip the optimizer step."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= scale_before
    if stepped:
        scheduler.step()
    return stepped


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mal_p, mal_y, bir_p, bir_y = [], [], [], []
    for x, mal, bir in loader:
        with autocast(device):
            lm, lb = model(to_device(x, device))
        mal_p.append(torch.sigmoid(lm.float()).cpu()), mal_y.append(mal)
        bir_p.append(lb.argmax(1).cpu()), bir_y.append(bir)
    mal_p, mal_y = torch.cat(mal_p).numpy(), torch.cat(mal_y).numpy()
    bir_p, bir_y = torch.cat(bir_p).numpy(), torch.cat(bir_y).numpy()
    m, b = mal_y >= 0, bir_y >= 0
    auc = roc_auc_score(mal_y[m], mal_p[m]) if len(np.unique(mal_y[m])) == 2 else float("nan")
    return {
        "birads_4_5_auc": auc,
        "birads_4_5_acc": float(((mal_p[m] > 0.5) == mal_y[m]).mean()) if m.any() else float("nan"),
        "birads_acc": float((bir_p[b] == bir_y[b]).mean()) if b.any() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="efficientnet_b3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0, help="limit rows per split (smoke tests)")
    args = p.parse_args()

    device = pick_device(args.device)
    df = pd.read_csv(LABELS_CSV, dtype={"malignant": "Int64", "birads_idx": "Int64"})
    df = df[(df.has_box == 1) & (df.malignant.notna() | df.birads_idx.notna())]
    df = df[[(DATA / p).exists() for p in df.image_path]]
    splits = {s: df[df.split == s] for s in ("train", "val")}
    if args.limit:
        splits = {s: d.head(args.limit) for s, d in splits.items()}
    print({s: len(d) for s, d in splits.items()}, "device:", device)

    train_dl = DataLoader(CropDataset(splits["train"], True), args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(CropDataset(splits["val"], False), args.batch, num_workers=args.workers)

    model = LesionClassifier(args.arch).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    best = -1.0
    MODELS.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, mal, bir in train_dl:
            x, mal, bir = to_device(x, device), mal.to(device), bir.to(device)
            with autocast(device):
                lm, lb = model(x)
            lm, lb = lm.float(), lb.float()
            m = mal >= 0
            loss = bce(lm[m], mal[m]) if m.any() else lm.sum() * 0
            if (bir >= 0).any():
                loss = loss + 0.5 * ce(lb, bir)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaled_optimizer_step(scaler, opt, sched)
            total += loss.item()
        metrics = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train_dl):.4f} {metrics}")
        score = metrics["birads_4_5_auc"] if not np.isnan(metrics["birads_4_5_auc"]) else metrics["birads_4_5_acc"]
        if score > best:
            best = score
            torch.save(
                {
                    "arch": args.arch,
                    "birads_classes": BIRADS_CLASSES,
                    "state_dict": model.state_dict(),
                    "val": metrics,
                },
                MODELS / "classifier.pt",
            )
            print(f"  saved {MODELS / 'classifier.pt'}")


if __name__ == "__main__":
    main()
