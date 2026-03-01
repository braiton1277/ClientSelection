import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE
from metrics import (
    flatten_params,
    flatten_grads,
    load_flat_params_,
    probing_loss_random_offset,
)


# ============================
# Server reference gradient
# ============================
def server_reference_grad(
    model: nn.Module, val_loader: DataLoader, batches: int = 10
) -> torch.Tensor:
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    for b, (x, y) in enumerate(val_loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    gref = flatten_grads(model).detach()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    return gref


# ============================
# Local training — steps curtos (para métricas)
# ============================
def local_train_delta(
    global_model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.005,
    steps: int = 10,
    momentum: float = 0.95,
    weight_decay: float = 1e-4,
    nesterov: bool = True,
) -> torch.Tensor:
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=(nesterov and momentum > 0.0),
    )
    w0 = flatten_params(model).clone()

    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    w1 = flatten_params(model)
    return (w1 - w0).detach()


# ============================
# Local training — epochs completos (para FedAvg)
# ============================
def local_train_selected(
    global_model: nn.Module,
    client_train_loaders: List[DataLoader],
    selected: List[int],
    lr: float = 0.005,
    epochs: int = 5,
    momentum: float = 0.95,
    weight_decay: float = 1e-4,
    nesterov: bool = True,
) -> List[Tuple[int, torch.Tensor]]:
    """
    Treina apenas os clientes selecionados com epochs completos.
    Retorna lista de (cid, delta) para usar no apply_fedavg.
    """
    results = []
    for cid in selected:
        model = copy.deepcopy(global_model).to(DEVICE)
        model.train()
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=(nesterov and momentum > 0.0),
        )
        w0 = flatten_params(model).clone()

        for _ in range(epochs):
            for x, y in client_train_loaders[cid]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()

        w1 = flatten_params(model)
        results.append((cid, (w1 - w0).detach()))

    return results


# ============================
# Compute deltas + projections + probing (usa steps curtos)
# ============================
def compute_deltas_proj_mom_probe_now_and_fo(
    model: nn.Module,
    client_train_loaders: List[DataLoader],
    client_eval_loaders: List[DataLoader],
    val_loader: DataLoader,
    local_lr: float,
    local_steps: int,
    probe_batches: int = 1,
    mom: Optional[torch.Tensor] = None,
    mom_beta: float = 0.90,
    round_seed: int = 0,
    momentum: float = 0.95,
    weight_decay: float = 1e-4,
    nesterov: bool = True,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    gref = server_reference_grad(model, val_loader, batches=10)

    if mom is None:
        mom = gref.detach().clone()
    else:
        mom = (mom_beta * mom) + ((1.0 - mom_beta) * gref.detach())

    desc_mom = (-mom).detach()
    desc_mom_norm = desc_mom / (desc_mom.norm() + 1e-12)

    desc_gref = (-gref).detach()
    desc_gref_norm = desc_gref / (desc_gref.norm() + 1e-12)

    deltas: List[torch.Tensor] = []
    probe_now: List[float] = []
    proj_mom: List[float] = []
    fo: List[float] = []

    for i, (tr_loader, ev_loader) in enumerate(zip(client_train_loaders, client_eval_loaders)):
        rng_i = np.random.RandomState(int(round_seed) + 1000 + i)
        probe_now.append(
            float(probing_loss_random_offset(model, ev_loader, batches=probe_batches, rng=rng_i))
        )

        dw = local_train_delta(
            model, tr_loader, lr=local_lr, steps=local_steps,
            momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
        )
        deltas.append(dw)

        proj_mom.append(float(torch.dot(dw, desc_mom_norm).item()))
        fo.append(float(torch.dot(dw, desc_gref_norm).item()))

    return (
        deltas,
        np.array(proj_mom, dtype=np.float32),
        np.array(probe_now, dtype=np.float32),
        np.array(fo, dtype=np.float32),
        mom.detach(),
    )


# ============================
# FedAvg aggregation
# ============================
# def apply_fedavg(
#     model: nn.Module,
#     deltas: List[Tuple[int, torch.Tensor]],
#     selected: List[int],
# ) -> None:
#     """
#     Aceita tanto lista de tensores (deltas de todos os clientes, indexada por cid)
#     quanto lista de tuplas (cid, delta) retornada por local_train_selected.
#     """
#     w = flatten_params(model).clone()
# 
#     # detecta formato: lista de tuplas (cid, delta) ou lista de tensores
#     if deltas and isinstance(deltas[0], tuple):
#         tensors = [d for _, d in deltas]
#     else:
#         tensors = [deltas[i] for i in selected]
# 
#     avg_dw = torch.stack(tensors, dim=0).mean(dim=0)
#     load_flat_params_(model, w + avg_dw)
# 



def apply_fedavg(
    model: nn.Module,
    deltas: List[Tuple[int, torch.Tensor]],
    selected: List[int],
) -> None:
    w = flatten_params(model).clone()

    if deltas and isinstance(deltas[0], tuple):
        tensors = [d for _, d in deltas]
    else:
        tensors = [deltas[i] for i in selected]

    # norm filtering — descarta deltas com norma > 2x a mediana
    norms = np.array([d.norm().item() for d in tensors])
    median_norm = np.median(norms)
    tensors = [d for d, n in zip(tensors, norms) if n < 2.0 * median_norm]

    if len(tensors) == 0:
        return  # segurança: se filtrou tudo, não atualiza

    avg_dw = torch.stack(tensors, dim=0).median(dim=0).values
    # clipping do delta agregado final
    max_norm = median_norm * 0.1
    avg_norm = avg_dw.norm().item()
    if avg_norm > max_norm:
        avg_dw = avg_dw * (max_norm / avg_norm)
    load_flat_params_(model, w + avg_dw)






















# ============================
# Staleness / streak tracking
# ============================
def update_staleness_streak(
    staleness: np.ndarray, streak: np.ndarray, selected: List[int]
) -> None:
    sel_mask = np.zeros(len(staleness), dtype=bool)
    sel_mask[selected] = True

    staleness[~sel_mask] += 1.0
    staleness[sel_mask] = 0.0

    streak[~sel_mask] = 0
    streak[sel_mask] += 1
