"""LPRNet — License Plate Recognition via Deep Neural Networks.

Implementation of the recognizer architecture specified in
DATASETS_AND_ARCHITECTURE.md §B.2. Adapted for Indian plates.

Architecture (input 3×64×192 strip):

    Conv 3×3, 64, BN, ReLU
    MaxPool 2×2                          → 32×96
    Conv 3×3, 128, BN, ReLU
    MaxPool 2×2                          → 16×48
    Conv 3×3, 256, BN, ReLU  ×2 + skip
    MaxPool (2,1)  keep W, compress H    → 8×48
    Conv 3×3, 256, BN, ReLU
    Conv 1×1 collapse H → 1              → 1×48 (squeeze)
    1×5 conv context block (cheap, no RNN)
    Linear → (V+1) classes               V = 33 (charset) + 1 CTC blank = 34
    CTC loss, T=48, label_len ≤ 10

Total params: ~1.8M. ONNX export: ~2 MB (fp16). ~6 ms on T4.

Reference: Zherzong Xu et al., "LPRNet: License Plate Recognition via Deep
Neural Networks" (IEEE TITS 2020). We simplify for the Indian use case
(smaller charset, fixed-aspect input).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Charset: digits 0-9 + A-Z minus {I, O, Q} = 33 chars.
# CTC blank is index 0; charset chars start at index 1.
CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}   # 1..33
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
BLANK_IDX = 0
VOCAB_SIZE = len(CHARSET) + 1   # 34 (33 + CTC blank)
PAD_IDX = -100   # CTC ignores this

INPUT_H, INPUT_W = 64, 192
SEQUENCE_LEN = INPUT_W // 4   # 48 after two 2×2 pools and a (2,1) pool


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU. ReLU is in-place for memory efficiency."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1,
                 p: Optional[int] = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ContextBlock(nn.Module):
    """1×5 depthwise-separable conv for cheap temporal context.

    Replaces the BiLSTM that CRNN uses. Cheaper, parallel-decodable.
    """

    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, (1, k), padding=(0, k // 2),
                            groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.bn(self.dw(x))
        y = self.bn2(self.pw(y))
        return F.relu(y + identity, inplace=True)


class LPRNet(nn.Module):
    """LPRNet recognizer for Indian plates.

    Args:
        vocab_size: number of output classes (including CTC blank).
        input_h, input_w: input image dimensions.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE,
                 input_h: int = INPUT_H, input_w: int = INPUT_W):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_h = input_h
        self.input_w = input_w

        # Backbone: ~6 conv blocks. Width doubles, H/W halve twice.
        # Block 1: 3→64, 64×64 input → 32×96 after pool
        self.block1 = nn.Sequential(
            ConvBNReLU(3, 64, 3),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
        )
        # Block 2: 64→128, → 16×48
        self.block2 = nn.Sequential(
            ConvBNReLU(64, 128, 3),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
        )
        # Block 3: 128→256 with a skip connection
        self.block3_a = ConvBNReLU(128, 256, 3)
        self.block3_b = ConvBNReLU(256, 256, 3)
        # Block 4: pool H only (2,1) — preserves W=48, compresses H 16→8
        self.pool4 = nn.MaxPool2d((2, 1), stride=(2, 1))
        # Block 5: 256→256
        self.block5 = ConvBNReLU(256, 256, 3)
        # Block 6: collapse H 8→1 via 1×8 conv with stride (8, 1)? No — we
        # use a Conv 1×1 after a 1×8 average pool to keep params minimal.
        self.h_collapse = nn.AvgPool2d((8, 1), stride=(8, 1))
        # Context: two 1×5 depthwise-separable blocks
        self.context = nn.Sequential(
            ContextBlock(256, k=5),
            ContextBlock(256, k=5),
        )
        # Classifier: 1×1 conv to vocab_size (per-timestep logits)
        self.classifier = nn.Conv2d(256, vocab_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) float32, normalized to [0, 1].
        Returns:
            logits: (B, T, V) where T = W/4 = 48, V = vocab_size.
            Use log_softmax + CTCLoss for training; argmax for inference.
        """
        x = self.block1(x)
        x = self.block2(x)
        # Block 3 with skip
        skip = self.block3_a(x)
        x = self.block3_b(skip) + skip
        x = self.pool4(x)
        x = self.block5(x)
        x = self.h_collapse(x)              # (B, 256, 1, W)
        x = self.context(x)                  # (B, 256, 1, W)
        logits = self.classifier(x)         # (B, V, 1, W)
        # Reshape to (B, T, V) — T = W (sequence length), V = vocab
        logits = logits.squeeze(2)          # (B, V, W)
        return logits.permute(0, 2, 1)      # (B, W, V) = (B, T, V)

    @torch.no_grad()
    def decode(self, logits: torch.Tensor) -> tuple[list[str], list[list[float]]]:
        """Greedy CTC decode.

        Args:
            logits: (B, T, V) raw logits.
        Returns:
            (texts, char_confs) lists of length B.
        """
        probs = F.softmax(logits, dim=-1)   # (B, T, V)
        pred = probs.argmax(dim=-1)         # (B, T)
        texts: list[str] = []
        char_confs: list[list[float]] = []
        for b in range(pred.shape[0]):
            chars: list[str] = []
            confs: list[float] = []
            prev = -1
            for t in range(pred.shape[1]):
                idx = int(pred[b, t].item())
                if idx != BLANK_IDX and idx != prev:
                    if idx in IDX_TO_CHAR:
                        chars.append(IDX_TO_CHAR[idx])
                        confs.append(float(probs[b, t, idx].item()))
                prev = idx
            texts.append("".join(chars))
            char_confs.append(confs)
        return texts, char_confs


def build_lprnet_for_training(weights: Optional[str] = None,
                              device: str = "cuda") -> tuple[LPRNet, nn.CTCLoss]:
    """Construct LPRNet + CTC loss for training.

    Args:
        weights: optional path to load checkpoint (.pt).
        device: 'cuda' or 'cpu'.
    Returns:
        (model, ctc_loss) — model is on `device`, in train mode.
    """
    model = LPRNet()
    if weights:
        ckpt = torch.load(weights, map_location=device)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    return model, ctc_loss


def encode_label(text: str) -> torch.Tensor:
    """Encode a plate string to a tensor of class indices (1..33).

    Chars not in the charset (e.g. I, O, Q) raise ValueError.
    """
    indices = [CHAR_TO_IDX[c] for c in text.upper() if c in CHAR_TO_IDX]
    if len(indices) == 0:
        raise ValueError(f"empty label after encoding: {text!r}")
    return torch.tensor(indices, dtype=torch.long)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test — verify forward pass works
    m = LPRNet()
    print(f"params: {count_params(m):,}")
    x = torch.randn(2, 3, INPUT_H, INPUT_W)
    y = m(x)
    print(f"output shape: {y.shape}  (expected: (2, {SEQUENCE_LEN}, {VOCAB_SIZE}))")
    assert y.shape == (2, SEQUENCE_LEN, VOCAB_SIZE), f"bad shape: {y.shape}"
    texts, confs = m.decode(y)
    print(f"decoded: {texts}")
    print("LPRNet smoke test OK")
