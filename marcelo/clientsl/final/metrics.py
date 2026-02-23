from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE


# ============================
# Flatten / load params
# ============================
def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def flatten_grads(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads)


def load_flat_params_(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view_as(p))
        offset += n


# ============================
# Evaluation
# ============================
@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, max_batches: int = 20) -> float:
    model.eval()
    total = 0.0
    n = 0
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, max_batches: int = 80) -> float:
    model.eval()
    correct = 0
    total = 0
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# ============================
# Probing loss
# ============================
@torch.no_grad()
def probing_loss(model: nn.Module, loader: DataLoader, batches: int = 1) -> float:
    model.eval()
    tot = 0.0
    n = 0
    for b, (x, y) in enumerate(loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        tot += loss.item()
        n += 1
    return tot / max(1, n)


@torch.no_grad()
def probing_loss_random_offset(
    model: nn.Module,
    loader: DataLoader,
    batches: int = 1,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Probing com offset aleatório de batches, evitando sempre avaliar os mesmos dados."""
    model.eval()
    if rng is None:
        rng = np.random.RandomState(0)

    try:
        n_total_batches = len(loader)
    except TypeError:
        n_total_batches = 0

    if n_total_batches <= 0:
        return probing_loss(model, loader, batches=batches)

    b = max(1, int(batches))
    max_start = max(0, n_total_batches - b)
    start = int(rng.randint(0, max_start + 1))

    tot = 0.0
    n = 0
    for bi, (x, y) in enumerate(loader):
        if bi < start:
            continue
        if bi >= start + b:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        tot += loss.item()
        n += 1

    return tot / max(1, n)


# ============================
# Gini coefficient
# ============================
def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    g = (n + 1.0 - 2.0 * np.sum(cum) / cum[-1]) / n
    return float(g)


# ============================
# Reward helper
# ============================
def windowed_reward(loss_history: list, new_loss: float, W: int = 5) -> float:
    if len(loss_history) == 0:
        base = new_loss
    else:
        w = min(W, len(loss_history))
        base = float(np.mean(loss_history[-w:]))
    raw = float(base - new_loss)
    denom = float(abs(base) + 1e-6)
    return raw / denom


def dynamic_batch_size(buf_n: int, base: int = 64, max_bs: int = 256, ratio: int = 4) -> int:
    bs = int(base)
    while (bs < max_bs) and (buf_n >= ratio * bs):
        bs *= 2
    return min(bs, max_bs)
