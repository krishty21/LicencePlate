"""LPRNet trainer — pretrain on synthetic, fine-tune on real Indian crops.

Implements the recipe from DATASETS_AND_ARCHITECTURE.md §B.2 / configs/ocr.yaml.

Stage 1 (pretrain):
    data: data/synthetic_indian_plates
    epochs: 10, bs 256, AdamW lr 1e-3, cosine schedule

Stage 2 (fine-tune):
    data: data/indian_plate_crops_real
    epochs: 5, bs 64, AdamW lr 1e-4
    oversample hard conditions ×5

Augmentation: perspective warp ±15°, motion blur, gamma 0.4–1.6,
JPEG q40, downscale to 24px then upsample (tiny-plate simulator).
"""
from __future__ import annotations

import csv
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .lprnet import (LPRNet, build_lprnet_for_training, encode_label,
                     INPUT_H, INPUT_W, VOCAB_SIZE, BLANK_IDX, CHARSET,
                     count_params)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters — mirrors configs/ocr.yaml train_lprnet."""

    # Stage 1 (pretrain)
    pretrain_data: Path = Path("data/synthetic_indian_plates")
    pretrain_epochs: int = 10
    pretrain_batch_size: int = 256
    pretrain_lr: float = 1e-3

    # Stage 2 (fine-tune)
    finetune_data: Optional[Path] = None
    finetune_epochs: int = 5
    finetune_batch_size: int = 64
    finetune_lr: float = 1e-4
    oversample_hard: int = 5

    # Common
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    output_dir: Path = Path("output/lprnet")
    log_every: int = 50
    save_every: int = 1   # save checkpoint every epoch

    # Augmentation
    aug_perspective_deg: float = 15.0
    aug_motion_blur_kernel_max: int = 5
    aug_gamma_range: tuple[float, float] = (0.4, 1.6)
    aug_jpeg_quality: int = 40
    aug_downscale_to_px: int = 24
    aug_prob: float = 0.5   # per-augmentation probability


class PlateDataset(Dataset):
    """Plate image + text dataset.

    Expected layout:
        root/images/000001.jpg
        root/labels.csv (filename, text, kind)
    """

    def __init__(self, root: Path, cfg: TrainConfig, train: bool = True):
        self.root = root
        self.cfg = cfg
        self.train = train
        self.images_dir = root / "images"
        labels_csv = root / "labels.csv"
        self.samples: list[tuple[Path, str]] = []
        if not labels_csv.exists():
            logger.warning("no labels.csv at %s", labels_csv)
            return
        with labels_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                text = row.get("text", "").upper()
                if not text:
                    continue
                # Skip text with chars outside our charset
                if any(c not in CHARSET for c in text):
                    continue
                img_path = self.images_dir / fname
                if img_path.exists():
                    self.samples.append((img_path, text))
        # Oversample hard conditions for fine-tune stage
        if train and cfg.oversample_hard > 1 and len(self.samples) > 0:
            self.samples = self.samples * cfg.oversample_hard
            random.shuffle(self.samples)
        logger.info("loaded %d samples from %s (train=%s, oversample=%s)",
                    len(self.samples), root, train,
                    cfg.oversample_hard if train else 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        import cv2
        img_path, text = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return a blank image + empty label if file is corrupted
            img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            text = ""
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        if self.train:
            img = self._augment(img)
        # BGR → RGB, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        if text:
            label = encode_label(text)
        else:
            label = torch.tensor([1], dtype=torch.long)   # placeholder
        return img, label, len(label)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        import cv2
        rng = random
        # Perspective warp
        if rng.random() < self.cfg.aug_prob:
            h, w = img.shape[:2]
            max_shift = 0.15
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            dst = src + np.array([
                [rng.uniform(-max_shift, max_shift) * w,
                 rng.uniform(-max_shift, max_shift) * h]] * 4,
                dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h))
        # Motion blur
        if rng.random() < self.cfg.aug_prob:
            k = rng.choice([3, 5, 7])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
        # Gamma
        if rng.random() < self.cfg.aug_prob:
            gamma = rng.uniform(*self.cfg.aug_gamma_range)
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
        # JPEG compress
        if rng.random() < self.cfg.aug_prob:
            q = self.cfg.aug_jpeg_quality
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # Tiny-plate simulator — downscale to aug_downscale_to_px wide then up
        if rng.random() < self.cfg.aug_prob * 0.5:
            h, w = img.shape[:2]
            small_w = max(8, self.cfg.aug_downscale_to_px)
            small_h = max(4, int(h * small_w / w))
            small = cv2.resize(img, (small_w, small_h),
                                interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return img


def collate_fn(batch):
    """Collate variable-length labels. Pads to max label length in batch."""
    imgs = torch.stack([b[0] for b in batch], 0)
    labels = [b[1] for b in batch]
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    # Concatenate labels (CTCLoss expects flat tensor)
    concatenated = torch.cat(labels)
    return imgs, concatenated, lengths


def cosine_lr(step: int, total: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate schedule."""
    if total <= 0:
        return base_lr
    progress = min(1.0, step / total)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch(model: LPRNet, ctc_loss: nn.CTCLoss, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, device: str,
                    epoch: int, base_lr: float, total_epochs: int,
                    log_every: int = 50) -> dict:
    """Train one epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    n_correct = 0   # exact-string matches
    n_total = 0
    steps = len(loader)
    for step, (imgs, labels, lengths) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        # LR schedule
        global_step = epoch * steps + step
        total_steps = total_epochs * steps
        lr = cosine_lr(global_step, total_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        # Forward
        logits = model(imgs)                    # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # type: ignore
        # CTC expects (T, B, V)
        log_probs_t = log_probs.permute(1, 0, 2)
        input_lengths = torch.full(
            (imgs.shape[0],), log_probs_t.shape[0],
            dtype=torch.long, device=device,
        )
        loss = ctc_loss(log_probs_t, labels, input_lengths, lengths)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        # Decode for accuracy
        with torch.no_grad():
            texts, _ = model.decode(logits)
            for i, t in enumerate(texts):
                # Recover the i-th label
                start = lengths[:i].sum().item()
                end = start + lengths[i].item()
                gt = "".join(CHARSET[c - 1] for c in labels[start:end].tolist()
                             if 0 < c <= len(CHARSET))
                if t == gt:
                    n_correct += 1
                n_total += 1
        total_loss += loss.item()
        if (step + 1) % log_every == 0:
            acc = n_correct / max(n_total, 1)
            logger.info(
                "epoch %d step %d/%d loss=%.4f lr=%.5f acc=%.3f",
                epoch, step + 1, steps, loss.item(), lr, acc,
            )
    return {
        "loss": total_loss / max(steps, 1),
        "accuracy": n_correct / max(n_total, 1),
    }


import torch.nn.functional as F   # late import to keep top of file clean


def train(cfg: TrainConfig) -> Path:
    """Run pretrain → fine-tune training. Returns path to best checkpoint."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, ctc_loss = build_lprnet_for_training(device=cfg.device)
    logger.info("LPRNet params: %s", f"{count_params(model):,}")
    best_acc = 0.0
    best_path = cfg.output_dir / "best.pt"
    # Stage 1: pretrain on synthetic
    if cfg.pretrain_data.exists():
        logger.info("=== Stage 1: pretrain on %s ===", cfg.pretrain_data)
        ds = PlateDataset(cfg.pretrain_data, cfg, train=True)
        if len(ds) > 0:
            loader = DataLoader(
                ds, batch_size=cfg.pretrain_batch_size, shuffle=True,
                num_workers=cfg.num_workers, collate_fn=collate_fn,
                pin_memory=True, drop_last=True,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.pretrain_lr, weight_decay=1e-4)
            for epoch in range(cfg.pretrain_epochs):
                metrics = train_one_epoch(
                    model, ctc_loss, loader, optimizer, cfg.device,
                    epoch, cfg.pretrain_lr, cfg.pretrain_epochs,
                    log_every=cfg.log_every,
                )
                logger.info("pretrain epoch %d: loss=%.4f acc=%.3f",
                            epoch, metrics["loss"], metrics["accuracy"])
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    torch.save({"model": model.state_dict(),
                                "epoch": epoch,
                                "accuracy": best_acc,
                                "stage": "pretrain"},
                               best_path)
                # Always save latest
                torch.save({"model": model.state_dict(), "epoch": epoch},
                           cfg.output_dir / "latest.pt")
    else:
        logger.warning("pretrain data not found: %s", cfg.pretrain_data)
    # Stage 2: fine-tune on real Indian crops
    if cfg.finetune_data and cfg.finetune_data.exists():
        logger.info("=== Stage 2: fine-tune on %s ===", cfg.finetune_data)
        ds = PlateDataset(cfg.finetune_data, cfg, train=True)
        if len(ds) > 0:
            loader = DataLoader(
                ds, batch_size=cfg.finetune_batch_size, shuffle=True,
                num_workers=cfg.num_workers, collate_fn=collate_fn,
                pin_memory=True, drop_last=True,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.finetune_lr, weight_decay=1e-4)
            for epoch in range(cfg.finetune_epochs):
                metrics = train_one_epoch(
                    model, ctc_loss, loader, optimizer, cfg.device,
                    epoch, cfg.finetune_lr, cfg.finetune_epochs,
                    log_every=cfg.log_every,
                )
                logger.info("finetune epoch %d: loss=%.4f acc=%.3f",
                            epoch, metrics["loss"], metrics["accuracy"])
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    torch.save({"model": model.state_dict(),
                                "epoch": epoch,
                                "accuracy": best_acc,
                                "stage": "finetune"},
                               best_path)
                torch.save({"model": model.state_dict(), "epoch": epoch},
                           cfg.output_dir / "latest.pt")
    elif cfg.finetune_data:
        logger.warning("finetune data not found: %s", cfg.finetune_data)
    logger.info("best accuracy: %.3f — saved to %s", best_acc, best_path)
    return best_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Train LPRNet")
    p.add_argument("--pretrain-data", type=Path,
                   default=Path("data/synthetic_indian_plates"))
    p.add_argument("--finetune-data", type=Path, default=None)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("output/lprnet"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TrainConfig(
        pretrain_data=args.pretrain_data,
        finetune_data=args.finetune_data,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_batch_size=args.batch_size,
        finetune_batch_size=max(16, args.batch_size // 4),
        device=args.device,
        output_dir=args.out,
        seed=args.seed,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
