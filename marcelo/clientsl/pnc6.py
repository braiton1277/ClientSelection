### validaçao no cliente
# fl_vdn_cifar10_pool50_k15_RANDOM_vs_VDN_TARGETED_PROJ_MOM_FO_PRINT.py
# =================================================================================
# Versão SEM DRIFT (removido 100%: grafo, detector, fork/reset).
#
# Mantém:
# - ataque TARGETED determinístico por amostra (switchable + attack_rate por cliente)
# - RANDOM vs VDN (um único track VDN)
# - print TODO round para os CLIENTES SELECIONADOS:
#     - cid, ATTACKER/HONEST
#     - estado usado pelo agente
#     - Q(a=0) e Q(a=1)
# - JSON final salvo no disco
#
# NOVO:
# - A cada N rodadas (default=20): imprime a lista (para TODOS os clientes):
#     - adv = (Q1 - Q0)
#     - FO (first-order credit) = <dw_i, -gref/||gref||>
#   (apenas para LOG; não afeta seleção nem treino)
#
# Estado VDN (d=5):
#   [bias, proj_mom, probe_loss_now, staleness_norm, streak_norm]
# onde:
#   proj_mom = <dw_i, -m/||m||> , m = beta*m_prev + (1-beta)*gref
#
# =================================================================================

import copy
import json
import random
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ============================
# Logging helper
# ============================
def log_step(msg: str):
    print(msg, flush=True)


# ============================
# Reproducibility (global)
# ============================
SEED = 2048
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ============================
# Small helpers for metrics
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
# Model (CIFAR-10 CNN)
# ============================
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32->16->8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8->4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================
# Flatten utils
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
# Eval + probing
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


# ============================
# Server reference gradient (explicit)
# ============================
def server_reference_grad(model: nn.Module, val_loader: DataLoader, batches: int = 10) -> torch.Tensor:
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
# Local training delta
# ============================
def local_train_delta(
    global_model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.01,
    steps: int = 10,
) -> torch.Tensor:
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
    dw = (w1 - w0).detach()
    return dw


# ============================
# Switchable deterministic TARGETED label flipping
# ============================
class SwitchableTargetedLabelFlipSubset(Dataset):
    """
    - Pré-computa por amostra:
        u[i] ~ U(0,1)
        flipped_label[i] via um mapeamento fixo (targeted)
    - Flip em tempo de execução:
        se enabled and u[i] < attack_rate
    - Determinístico por amostra, e attack_rate controla a fração atacada.
    """
    def __init__(
        self,
        base_ds,
        indices,
        n_classes: int = 10,
        seed: int = 0,
        enabled: bool = False,
        attack_rate: float = 0.0,
        target_map: Optional[Dict[int, int]] = None,
        only_map_classes: bool = True,
    ):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.n_classes = int(n_classes)
        self.enabled = bool(enabled)
        self.attack_rate = float(attack_rate)
        self.only_map_classes = bool(only_map_classes)

        # Default: swaps "pareados" (targeted e consistente)
        if target_map is None:
            target_map = {
                0: 8,  # airplane -> ship
                8: 0,  # ship -> airplane
                1: 9,  # automobile -> truck
                9: 1,  # truck -> automobile
                3: 5,  # cat -> dog
                5: 3,  # dog -> cat
                4: 7,  # deer -> horse
                7: 4,  # horse -> deer
                2: 6,  # bird -> frog
                6: 2,  # frog -> bird
            }
        self.target_map = {int(k): int(v) for k, v in target_map.items()}

        rng = np.random.RandomState(seed)
        self.u = rng.rand(len(self.indices)).astype(np.float32)

        self.flipped_label = np.zeros(len(self.indices), dtype=np.int64)
        for i, idx in enumerate(self.indices):
            _, y = self.base_ds[idx]
            y = int(y)

            if y in self.target_map:
                y_new = int(self.target_map[y])
            else:
                if self.only_map_classes:
                    y_new = y
                else:
                    y_new = rng.randint(0, self.n_classes - 1)
                    if y_new >= y:
                        y_new += 1

            y_new = int(np.clip(y_new, 0, self.n_classes - 1))
            if y_new == y:
                y_new = (y + 1) % self.n_classes

            self.flipped_label[i] = y_new

    def set_attack(self, enabled: bool, rate: float):
        self.enabled = bool(enabled)
        self.attack_rate = float(rate)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base_ds[self.indices[i]]
        y = int(y)
        if self.enabled and (self.attack_rate > 0.0) and (self.u[i] < self.attack_rate):
            y = int(self.flipped_label[i])
        return x, y

class CleanViewOfSwitchable(Dataset):
    """Mesmo cliente (mesmos índices), mas SEM flip: usa rótulo original do base_ds."""
    def __init__(self, flip_ds: SwitchableTargetedLabelFlipSubset):
        self.base_ds = flip_ds.base_ds
        self.indices = flip_ds.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[self.indices[i]]


# ============================
# Server balanced validation (from TRAIN)
# ============================
def make_server_val_balanced(ds, per_class: int = 200, n_classes: int = 10, seed: int = 0) -> List[int]:
    rng = np.random.RandomState(seed)
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        label_to_idxs[int(y)].append(idx)

    val = []
    for y in range(n_classes):
        idxs = label_to_idxs[y]
        rng.shuffle(idxs)
        val.extend(idxs[:per_class])

    rng.shuffle(val)
    return val


# ============================
# Dirichlet non-IID split
# ============================
#def make_clients_dirichlet_indices(
 #   train_ds,
 #   n_clients: int = 50,
 #   alpha: float = 0.3,
 #   seed: int = 123,
 #   n_classes: int = 10
#) -> List[List[int]]:
 #   rng = np.random.RandomState(seed)
#
  #  label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
  #  for idx in range(len(train_ds)):
  #      _, y = train_ds[idx]
 #       label_to_idxs[int(y)].append(idx)
#
 #   for y in range(n_classes):
 #       rng.shuffle(label_to_idxs[y])
#
 #   clients = [[] for _ in range(n_clients)]
#
   # for y in range(n_classes):
   #     idxs = label_to_idxs[y]
   #     props = rng.dirichlet(alpha * np.ones(n_clients))
  #      counts = (props * len(idxs)).astype(int)
#
  #      diff = len(idxs) - counts.sum()
 #       if diff > 0:
 #           for j in rng.choice(n_clients, size=diff, replace=True):
    #            counts[j] += 1
   #     elif diff < 0:
  #          for j in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
 #               counts[j] -= 1
#
      #  start = 0
     #   for cid in range(n_clients):
    #        c = counts[cid]
   #        if c > 0:
  #              clients[cid].extend(idxs[start:start + c])
 #               start += c
#
  #  for cid in range(n_clients):
 #       rng.shuffle(clients[cid])
#
    #return clients




def make_clients_dirichlet_indices_minmax(
    train_ds,
    n_clients: int = 50,
    alpha: float = 0.3,
    seed: int = 123,
    n_classes: int = 10,
    min_size: int = 500,
    max_size: int = 2000,
    max_tries: int = 10_000,
):
    rng = np.random.RandomState(seed)

    # pega índices por classe
    label_to_idxs = {i: [] for i in range(n_classes)}
    for idx in range(len(train_ds)):
        _, y = train_ds[idx]
        label_to_idxs[int(y)].append(idx)

    for y in range(n_classes):
        rng.shuffle(label_to_idxs[y])

    for _ in range(max_tries):
        clients = [[] for _ in range(n_clients)]

        for y in range(n_classes):
            idxs = label_to_idxs[y].copy()
            rng.shuffle(idxs)

            props = rng.dirichlet(alpha * np.ones(n_clients))
            counts = (props * len(idxs)).astype(int)

            diff = len(idxs) - counts.sum()
            if diff > 0:
                for j in rng.choice(n_clients, size=diff, replace=True):
                    counts[j] += 1
            elif diff < 0:
                pos = np.where(counts > 0)[0]
                for j in rng.choice(pos, size=-diff, replace=True):
                    counts[j] -= 1

            start = 0
            for cid in range(n_clients):
                c = counts[cid]
                if c > 0:
                    clients[cid].extend(idxs[start:start + c])
                    start += c

        sizes = np.array([len(c) for c in clients], dtype=int)
        if sizes.min() >= min_size and sizes.max() <= max_size:
            for cid in range(n_clients):
                rng.shuffle(clients[cid])
            return clients

    raise RuntimeError(
        f"Não achei split com min_size={min_size}, max_size={max_size} "
        f"em {max_tries} tentativas. (alpha={alpha})"
    )













# ============================
# Staleness / streak updates
# ============================
def update_staleness_streak(staleness: np.ndarray, streak: np.ndarray, selected: List[int]):
    n = len(staleness)
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    staleness[~sel_mask] += 1.0
    staleness[sel_mask] = 0.0

    streak[~sel_mask] = 0
    streak[sel_mask] += 1


# ============================
# Compute deltas + projection on global momentum + probe_now
# + FO (para logging) + update momentum
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
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    # gref explícito no server_val
    gref = server_reference_grad(model, val_loader, batches=10)

    # momento do gradiente global
    if mom is None:
        mom = gref.detach().clone()
    else:
        mom = (mom_beta * mom) + ((1.0 - mom_beta) * gref.detach())

    # direção de descida com momento (para proj_mom)
    desc_mom = (-mom).detach()
    desc_mom_norm = desc_mom / (desc_mom.norm() + 1e-12)

    # direção de descida SEM momento (para FO logging)
    desc_gref = (-gref).detach()
    desc_gref_norm = desc_gref / (desc_gref.norm() + 1e-12)

    deltas: List[torch.Tensor] = []
    probe_now: List[float] = []
    proj_mom: List[float] = []
    fo: List[float] = []

    for tr_loader, ev_loader in zip(client_train_loaders, client_eval_loaders):
        probe_now.append(float(probing_loss(model, ev_loader, batches=probe_batches)))

        dw = local_train_delta(model, tr_loader, lr=local_lr, steps=local_steps)
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
# Apply FedAvg
# ============================
def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor], selected: List[int]):
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


# ============================
# Reward helpers (interno p/ RL)
# ============================
def windowed_reward(loss_history: List[float], new_loss: float, W: int = 5) -> float:
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
    max_bs = int(max_bs)
    ratio = int(ratio)
    while (bs < max_bs) and (buf_n >= ratio * bs):
        bs *= 2
    return min(bs, max_bs)


# ============================
# VDN state
# Estado por cliente:
#   [bias, proj_mom, probe_loss_now, staleness_norm, streak_norm]
# d_in = 5
# ============================
def build_context_matrix_vdn(
    projection_mom: np.ndarray,
    probe_now: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
) -> np.ndarray:
    proj = projection_mom.astype(np.float32)
    probe = probe_now.astype(np.float32)

    s = staleness.astype(np.float32)
    smax = float(s.max() + 1e-6)
    sn = (s / smax).astype(np.float32)

    cap = 5.0
    t = streak.astype(np.float32)
    tn = np.clip(t / cap, 0.0, 1.0).astype(np.float32)

    bias = np.ones((proj.shape[0],), dtype=np.float32)
    X = np.stack([bias, proj, probe, sn, tn], axis=1).astype(np.float32)  # (N,5)
    return X


# ============================
# PER buffer (joint transitions)
# ============================
class PrioritizedReplayJoint:
    def __init__(self, capacity: int, n_agents: int, d_in: int,
                 alpha: float = 0.6, eps: float = 1e-3, seed: int = 0):
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.obs  = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.act  = np.zeros((capacity, n_agents), dtype=np.uint8)
        self.r    = np.zeros((capacity,), dtype=np.float32)
        self.obs2 = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.p = np.zeros((capacity,), dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.max_p = 1.0

        self.rng = np.random.default_rng(int(seed))

    def add(self, obs, act, r, obs2, done: bool):
        self.obs[self.ptr] = obs.astype(np.float32)
        self.act[self.ptr] = act.astype(np.uint8)
        self.r[self.ptr] = float(r)
        self.obs2[self.ptr] = obs2.astype(np.float32)
        self.done[self.ptr] = 1.0 if done else 0.0

        self.p[self.ptr] = self.max_p
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        bs = min(int(batch_size), self.n)
        assert bs > 0

        pri = self.p[:self.n].astype(np.float64)
        probs = (pri + self.eps) ** self.alpha
        s = probs.sum()
        if s <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / s

        idx = self.rng.choice(self.n, size=bs, replace=False, p=probs)
        w = (self.n * probs[idx]) ** (-beta)
        w = w / (w.max() + 1e-12)

        return (
            self.obs[idx], self.act[idx], self.r[idx], self.obs2[idx], self.done[idx],
            idx.astype(np.int64), w.astype(np.float32)
        )

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.p[idx] = td_abs + self.eps
        self.max_p = float(max(self.max_p, float(td_abs.max(initial=0.0))))


# ============================
# VDN (MLP) + top-K + Double-DQN
# ============================
class AgentMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VDNSelector:
    def __init__(
        self,
        n_agents: int,
        d_in: int,
        k_select: int,
        hidden: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.90,
        grad_clip: float = 1.0,
        target_sync_every: int = 20,
        buf_size: int = 20000,
        batch_size: int = 128,
        train_steps: int = 20,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 4000,
        per_eps: float = 1e-3,
        double_dqn: bool = True,
        seed: int = 0,
    ):
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.k_select = int(k_select)
        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)

        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)

        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = int(per_beta_steps)

        self.hidden = int(hidden)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.per_alpha = float(per_alpha)
        self.per_eps = float(per_eps)

        self.q = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)

        self.buf_size = int(buf_size)
        self.buf = PrioritizedReplayJoint(
            capacity=buf_size, n_agents=self.n_agents, d_in=self.d_in,
            alpha=per_alpha, eps=per_eps, seed=int(seed) + 12345
        )

        self._train_calls = 0
        self.total_updates = 0
        self.total_samples_drawn = 0

        self.seed = int(seed)
        self.py_rng = random.Random(int(seed) + 777)
        self.np_rng = np.random.default_rng(int(seed) + 999)

    def _beta(self) -> float:
        t = min(self._train_calls, self.per_beta_steps)
        frac = t / max(1, self.per_beta_steps)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    @torch.no_grad()
    def _q_all_agents(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)  # (N,d)
        q = self.q(x)                                              # (N,2)
        return q.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        q = self.q(x)
        return q.detach().cpu().numpy().astype(np.float32)

    def select_topk_actions(
        self,
        obs: np.ndarray,
        eps: float = 0.15,
        swap_m: int = 2,
        force_random: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        n = obs.shape[0]
        K = min(self.k_select, n)

        q = self._q_all_agents(obs)  # (N,2)

        if force_random:
            sel = self.py_rng.sample(range(n), K)
            a = np.zeros(n, dtype=np.uint8)
            a[sel] = 1
            return a, sel

        adv = q[:, 1] - q[:, 0]
        sel = np.argsort(adv)[::-1][:K].tolist()

        if swap_m > 0 and (self.np_rng.random() < eps):
            swap_m = min(swap_m, K)
            sel_set = set(sel)
            not_sel = [i for i in range(n) if i not in sel_set]
            if len(not_sel) > 0:
                out = self.py_rng.sample(sel, swap_m)
                inn = self.py_rng.sample(not_sel, min(swap_m, len(not_sel)))
                sel2 = sel.copy()
                for o, i_new in zip(out, inn):
                    sel2[sel2.index(o)] = i_new
                sel2 = list(dict.fromkeys(sel2))
                while len(sel2) < K:
                    cand = self.py_rng.randrange(n)
                    if cand not in sel2:
                        sel2.append(cand)
                sel = sel2[:K]

        a = np.zeros(n, dtype=np.uint8)
        a[sel] = 1
        return a, sel

    def add_transition(self, obs, act, r, obs2, done: bool):
        self.buf.add(obs=obs, act=act, r=r, obs2=obs2, done=done)

    def train(self, batch_size: Optional[int] = None, train_steps: Optional[int] = None) -> Optional[float]:
        bs_req = int(batch_size) if batch_size is not None else self.batch_size
        steps = int(train_steps) if train_steps is not None else self.train_steps

        if self.buf.n < max(32, bs_req):
            return None

        beta = self._beta()
        self.q.train()

        losses = []
        for _ in range(steps):
            ob, ac, rw, ob2, dn, idx, w_is = self.buf.sample(batch_size=bs_req, beta=beta)
            B = ob.shape[0]
            self.total_samples_drawn += B

            obs  = torch.tensor(ob,  dtype=torch.float32, device=DEVICE)   # (B,N,D)
            act  = torch.tensor(ac,  dtype=torch.long,   device=DEVICE)    # (B,N)
            r    = torch.tensor(rw,  dtype=torch.float32, device=DEVICE)   # (B,)
            obs2 = torch.tensor(ob2, dtype=torch.float32, device=DEVICE)   # (B,N,D)
            done = torch.tensor(dn,  dtype=torch.float32, device=DEVICE)   # (B,)
            w    = torch.tensor(w_is, dtype=torch.float32, device=DEVICE)  # (B,)

            N = self.n_agents

            q_cur = self.q(obs.reshape(B * N, self.d_in)).reshape(B, N, 2)
            q_a = q_cur.gather(2, act.unsqueeze(2)).squeeze(2)
            q_tot = q_a.sum(dim=1)

            with torch.no_grad():
                q2_online = self.q(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)
                q2_tgt    = self.q_tgt(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)

                K = min(self.k_select, N)
                adv2 = (q2_online[:, :, 1] - q2_online[:, :, 0]) if self.double_dqn else (q2_tgt[:, :, 1] - q2_tgt[:, :, 0])
                top_idx = adv2.topk(K, dim=1).indices
                a2 = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
                a2.scatter_(1, top_idx, 1)

                q2_a = q2_tgt.gather(2, a2.unsqueeze(2)).squeeze(2)
                q2_tot = q2_a.sum(dim=1)
                y = r + (1.0 - done) * self.gamma * q2_tot

            td = (q_tot - y)
            td_abs = td.detach().abs().cpu().numpy().astype(np.float32)

            per_step = F.smooth_l1_loss(q_tot, y, reduction="none")
            loss = (w * per_step).mean()

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

            self.buf.update_priorities(idx, td_abs)

            losses.append(float(loss.item()))
            self.total_updates += 1

        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(np.mean(losses)) if losses else None


# ============================
# Experiment runner
# ============================
def run_experiment(
    rounds: int = 300,
    n_clients: int = 50,
    k_select: int = 15,
    dir_alpha: float = 0.3,

    # Ataque acumulativo
    initial_flip_fraction: float = 0.0,
    flip_add_fraction: float = 0.20,
    attack_rounds: List[int] = None,
    flip_rate_initial: float = 1.0,
    flip_rate_new_attack: float = 1.0,

    # Targeted attack options
    targeted_only_map_classes: bool = True,
    target_map: Optional[Dict[int, int]] = None,

    max_per_client: int = 2500,
    local_lr: float = 0.01,
    local_steps: int = 10,
    probe_batches: int = 5,

    # GLOBAL MOMENTUM
    mom_beta: float = 0.90,

    reward_window_W: int = 5,

    marl_eps: float = 0.15,
    marl_swap_m: int = 2,
    marl_lr: float = 1e-3,
    marl_gamma: float = 0.90,
    marl_hidden: int = 128,
    marl_target_sync_every: int = 20,

    warmup_transitions: int = 200,
    start_train_round: int = 100,
    updates_per_round: int = 50,
    train_every: int = 1,

    buf_size: int = 20000,
    batch_base: int = 64,
    batch_max: int = 256,
    batch_buffer_ratio: int = 4,

    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_steps: int = 4000,
    per_eps: float = 1e-3,

    val_shuffle: bool = False,
    val_per_class: int = 200,
    eval_max_batches: int = 20,
    print_every: int = 10,

    # NOVO: print (adv = Q1-Q0) e FO a cada N rounds
    print_advfo_every: int = 20,

    out_dir: str = ".",
):
    if attack_rounds is None:
        attack_rounds = [150]
    attack_rounds = sorted(list(set(int(x) for x in attack_rounds)))

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfm_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tfm_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # ---------- JSON log skeleton ----------
    run_id = uuid.uuid4().hex[:10]
    out_path = Path(out_dir) / f"results_random_vs_vdn_targeted_PROJMOM_FO_seed{SEED}_{run_id}.json"

    log = {
        "meta": {
            "run_id": run_id,
            "seed": int(SEED),
            "device": str(DEVICE),
            "rounds": int(rounds),
            "n_clients": int(n_clients),
            "k_select": int(k_select),
            "dir_alpha": float(dir_alpha),

            "initial_flip_fraction": float(initial_flip_fraction),
            "flip_add_fraction": float(flip_add_fraction),
            "attack_rounds": list(attack_rounds),
            "flip_rate_initial": float(flip_rate_initial),
            "flip_rate_new_attack": float(flip_rate_new_attack),

            "targeted_only_map_classes": bool(targeted_only_map_classes),
            "target_map": (target_map if target_map is not None else "default_pair_swaps"),

            "mom_beta": float(mom_beta),

            "buf_size": int(buf_size),
            "warmup_transitions": int(warmup_transitions),
            "start_train_round": int(start_train_round),
            "updates_per_round": int(updates_per_round),
            "train_every": int(train_every),

            "print_advfo_every": int(print_advfo_every),
        },
        "attack_schedule": [],
        "tracks": {
            "random": {
                "test_acc": [],
                "selection_count_total_per_client": [0] * int(n_clients),
                "selection_phases": [],
            },
            "vdn": {
                "test_acc": [],
                "selection_count_total_per_client": [0] * int(n_clients),
                "selection_phases": [],
            },
        },
    }

    def save_json_safely():
        for key in ["random", "vdn"]:
            cnt_total = np.array(log["tracks"][key]["selection_count_total_per_client"], dtype=np.int64)
            log["tracks"][key]["final_metrics"] = {
                "gini_selection_total": float(gini_coefficient(cnt_total)),
                "total_selections": int(cnt_total.sum()),
            }
            phases = log["tracks"][key]["selection_phases"]
            for ph in phases:
                if ph.get("end_round", None) is None:
                    ph["end_round"] = int(rounds)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"\n[JSON] salvo em: {str(out_path)}\n", flush=True)

    def start_new_phase(track_key: str, start_round: int, attacked_snapshot: List[int]):
        log["tracks"][track_key]["selection_phases"].append({
            "start_round": int(start_round),
            "end_round": None,
            "attacked_clients_snapshot": list(attacked_snapshot),
            "selection_count_per_client": [0] * int(n_clients),
        })

    def bump_counts(track_key: str, selected: List[int]):
        total_cnt = log["tracks"][track_key]["selection_count_total_per_client"]
        phases = log["tracks"][track_key]["selection_phases"]
        phase_cnt = phases[-1]["selection_count_per_client"]
        for i in selected:
            total_cnt[i] += 1
            phase_cnt[i] += 1



    # ---------- Data ----------
    log_step("Baixando/carregando CIFAR-10...")
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    log_step("Criando server_val balanceado (holdout do TRAIN) + train_pool (sem val)...")
    server_val_idxs = make_server_val_balanced(train_ds, per_class=val_per_class, n_classes=10, seed=SEED + 4242)
    server_val_set = set(server_val_idxs)

    all_train_idxs = np.arange(len(train_ds))
    train_pool_idxs = [int(i) for i in all_train_idxs if int(i) not in server_val_set]
    train_pool = Subset(train_ds, train_pool_idxs)

    g_val = torch.Generator()
    g_val.manual_seed(SEED + 123)

    val_loader = DataLoader(
        Subset(train_ds, server_val_idxs),
        batch_size=256,
        shuffle=val_shuffle,
        generator=g_val,
        worker_init_fn=seed_worker,
        num_workers=0,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    log_step(f"Gerando split Dirichlet (alpha={dir_alpha}) para {n_clients} clientes (em train_pool)...")
    target = len(train_pool) // n_clients  # tamanho médio esperado


    client_idxs = make_clients_dirichlet_indices_minmax(
        train_pool, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777, n_classes=10,min_size=int(0.2 * target),
    max_size=int(1.8 * target),
    )

    # ---------- Ataque acumulativo: inicial ----------
    n_init = int(round(initial_flip_fraction * n_clients))
    rng_init = np.random.RandomState(SEED + 999)
    attacked_set = set(rng_init.choice(np.arange(n_clients), size=n_init, replace=False).tolist()) if n_init > 0 else set()

    attack_rate_per_client = np.zeros(n_clients, dtype=np.float32)
    for cid in attacked_set:
        attack_rate_per_client[cid] = float(flip_rate_initial)

    # ---------- Client datasets/loaders ----------
    client_train_loaders: List[DataLoader] = []
    client_probe_loaders: List[DataLoader] = []
    client_val_loaders: List[DataLoader] = []
    client_sizes: List[int] = []
    switchable_ds: List[SwitchableTargetedLabelFlipSubset] = []

    g_train = torch.Generator()
    g_train.manual_seed(SEED + 10001)
    
    client_sizes_total: List[int] = []
    client_sizes_train: List[int] = []
    client_sizes_val: List[int] = []

    log_step("Criando loaders dos clientes + label flipping TARGETED determinístico (switchable, taxa ajustável)...")
    for cid, idxs in enumerate(client_idxs):
        if max_per_client is not None:
            idxs = idxs[:max_per_client]
        client_sizes.append(len(idxs))


        client_val_frac = 0.2

        ds_c = SwitchableTargetedLabelFlipSubset(
            base_ds=train_pool,
            indices=idxs,
            n_classes=10,
            seed=SEED + 1000 + cid,
            enabled=(cid in attacked_set),
            attack_rate=float(attack_rate_per_client[cid]),
            target_map=target_map,
            only_map_classes=targeted_only_map_classes,
            )
        switchable_ds.append(ds_c)

        # "visão limpa" do mesmo cliente (sem flip)
        ds_clean = CleanViewOfSwitchable(ds_c)

        # split por POSIÇÃO dentro do dataset do cliente (determinístico)
        L = len(ds_c)
        pos = np.arange(L)
        rng_split = np.random.RandomState(SEED + 2222 + cid)
        rng_split.shuffle(pos)

        n_val = max(1, int(round(client_val_frac * L)))
        n_val = min(n_val, L - 1) if L > 1 else 1  # garante que sobra treino

        val_pos = pos[:n_val].tolist()
        tr_pos  = pos[n_val:].tolist()

        client_sizes_total.append(int(L))
        client_sizes_train.append(int(len(tr_pos)))
        client_sizes_val.append(int(len(val_pos)))


        ds_tr = Subset(ds_c, tr_pos)          # TREINO: com flip (atacantes)
        ds_val = Subset(ds_clean, val_pos)    # VAL: sem flip (limpo)

        client_train_loaders.append(
            DataLoader(ds_tr, batch_size=64, shuffle=True, generator=g_train, worker_init_fn=seed_worker, num_workers=0)
            )


        # PROBING (com flip): use o dataset do cliente (ou ds_tr, se preferir)
        client_probe_loaders.append(
            DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=0)
            )



        client_val_loaders.append(
            DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=0)
            )








    # ===== LOG: tamanhos por cliente ===== 
    log["meta"]["client_sizes_total"] = list(map(int, client_sizes_total))
    log["meta"]["client_sizes_train"] = list(map(int, client_sizes_train))
    log["meta"]["client_sizes_val"]   = list(map(int, client_sizes_val))

    sizes = np.array(client_sizes_total, dtype=np.int32)
    print("\n[CLIENT DATA SIZES] cid | total | train | val")
    for cid in range(n_clients):
        print(f"  {cid:02d} | {client_sizes_total[cid]:4d} | {client_sizes_train[cid]:4d} | {client_sizes_val[cid]:4d}")

    print(
        f"\n[CLIENT SIZE STATS] total: "
        f"min={sizes.min()} | mean={sizes.mean():.1f} | max={sizes.max()} | "
        f"p10={int(np.percentile(sizes,10))} | p90={int(np.percentile(sizes,90))}\n"
        )






    # ---------- Models ----------
    base = SmallCNN().to(DEVICE)

    # RANDOM baseline
    model_rand = copy.deepcopy(base).to(DEVICE)
    rng_random_sel = random.Random(SEED + 424242)
    start_new_phase("random", 1, sorted(list(attacked_set)))

    # VDN
    model_vdn = copy.deepcopy(base).to(DEVICE)
    staleness_v = np.zeros(n_clients, dtype=np.float32)
    streak_v = np.zeros(n_clients, dtype=np.int32)
    loss_hist_v: List[float] = []
    pending_v: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    mom_v: Optional[torch.Tensor] = None  # momentum do gradiente global

    agent_v = VDNSelector(
        n_agents=n_clients,
        d_in=5,
        k_select=k_select,
        hidden=marl_hidden,
        lr=marl_lr,
        weight_decay=1e-4,
        gamma=marl_gamma,
        grad_clip=1.0,
        target_sync_every=marl_target_sync_every,
        buf_size=buf_size,
        batch_size=batch_base,
        train_steps=max(1, updates_per_round),
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        per_beta_steps=per_beta_steps,
        per_eps=per_eps,
        double_dqn=True,
        seed=SEED + 10,
    )

    start_new_phase("vdn", 1, sorted(list(attacked_set)))

    print(f"\nDEVICE={DEVICE}")
    print(f"POOL N_CLIENTS={n_clients} | K={k_select} | rounds={rounds} | dir_alpha={dir_alpha}")
    print(f"TRAIN: total={len(train_ds)} | server_val={len(server_val_idxs)} | train_pool={len(train_pool_idxs)}")
    print(f"TEST (intocado): {len(test_ds)}")
    print(f"Ataque acumulativo: init_frac={initial_flip_fraction} (n_init={n_init}) | add_frac={flip_add_fraction} | attack_rounds={attack_rounds}")
    print(f"Flip rates: initial={flip_rate_initial} | new_attack={flip_rate_new_attack}")
    print(f"TARGETED: only_map_classes={targeted_only_map_classes} | map={'default' if target_map is None else 'custom'}")
    print(f"Global momentum: beta={mom_beta}")
    print(f"Estado VDN: [bias, proj_mom, probe_now, staleness_n, streak_n] (d=5)")
    print(f"PRINT adv/FO: every {print_advfo_every} rounds (0/<=0 desliga)")
    print(f"Avg client size ~ {np.mean(client_sizes_total):.1f} samples")
    print(f"Tracks: [RANDOM] vs [VDN]\n")


    client_val_every = 10
    client_val_max_batches = 9999  # controla custo; aumente se quiser mais "completo"




    try:
        for t in range(1, rounds + 1):
            log_step(f"\n[round {t}/{rounds}] começando...")

            # ===== Ataque acumulativo =====
            if t in attack_rounds:
                n_add = int(round(flip_add_fraction * n_clients))
                candidates = [i for i in range(n_clients) if i not in attacked_set]
                rng_add = np.random.RandomState(SEED + 5000 + t)
                rng_add.shuffle(candidates)
                add_now = candidates[:min(n_add, len(candidates))]

                for cid in add_now:
                    attacked_set.add(cid)
                    attack_rate_per_client[cid] = float(flip_rate_new_attack)

                for cid, ds in enumerate(switchable_ds):
                    if cid in attacked_set:
                        ds.set_attack(True, float(attack_rate_per_client[cid]))
                    else:
                        ds.set_attack(False, 0.0)

                log["attack_schedule"].append({
                    "round": int(t),
                    "added_clients": list(map(int, add_now)),
                    "rate_for_added": float(flip_rate_new_attack),
                    "attacked_total_after": int(len(attacked_set)),
                })
                log_step(f"  >>> ATTACK ADD @ round {t}: adicionados {len(add_now)} novos atacados | attacked_total={len(attacked_set)}")

            attacked_snapshot = sorted(list(attacked_set))

            round_seed = SEED + 50000 + t
            g_train.manual_seed(round_seed)

            # ============================================================
            # TRACK A: RANDOM
            # ============================================================
            a_rand = eval_acc(model_rand, test_loader, max_batches=80)

            deltas_r, _, _, _, _ = compute_deltas_proj_mom_probe_now_and_fo(
                model_rand,
                client_train_loaders,
                client_probe_loaders,
                val_loader,
                local_lr,
                local_steps,
                probe_batches=probe_batches,
                mom=None,
                mom_beta=mom_beta,
            )

            K = min(k_select, n_clients)
            sel_r = rng_random_sel.sample(range(n_clients), K)
            apply_fedavg(model_rand, deltas_r, sel_r)

            bump_counts("random", sel_r)
            log["tracks"]["random"]["test_acc"].append(float(a_rand))

            # ============================================================
            # TRACK B: VDN
            # ============================================================
            acc_v = eval_acc(model_vdn, test_loader, max_batches=80)

            _l_before = eval_loss(model_vdn, val_loader, max_batches=eval_max_batches)

            deltas_v, proj_mom_v, probe_now_v, fo_v, mom_v = compute_deltas_proj_mom_probe_now_and_fo(
                model_vdn,
                client_train_loaders,
                client_probe_loaders,
                val_loader,
                local_lr,
                local_steps,
                probe_batches=probe_batches,
                mom=mom_v,
                mom_beta=mom_beta,
            )

            obs_v = build_context_matrix_vdn(
                projection_mom=proj_mom_v,
                probe_now=probe_now_v,
                staleness=staleness_v,
                streak=streak_v,
            )

            # adiciona transição pendente (S_{t-1} -> S_t)
            if pending_v is not None:
                o_prev, a_prev, r_prev = pending_v
                agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=obs_v, done=False)

            force_rand = (agent_v.buf.n < warmup_transitions)

            swap_eps = 0.20
            half = rounds *0.6
            if t<=half:
                swap_m_sched = 5
            else:
                swap_m_sched = 2


            act_v, sel_v = agent_v.select_topk_actions(
                obs=obs_v,
                eps=marl_eps,
                swap_m=swap_m_sched,
                force_random=force_rand,
            )

            # ===== PRINT estados dos selecionados + Q + atacante/honesto (NÃO MEXI) =====
            q_all = agent_v.q_values(obs_v)  # (N,2)
            print("\n[SELECTED DEBUG] cid | flag | state=[bias, proj_mom, probe_now, stale_n, streak_n] | Q0 Q1")
            for cid in sel_v:
                flag = "ATTACKER" if cid in attacked_set else "HONEST"
                st = obs_v[cid]
                q0, q1 = float(q_all[cid, 0]), float(q_all[cid, 1])
                print(
                    f"  {cid:02d} | {flag:8s} | "
                    f"[{st[0]:.3f}, {st[1]:+.4f}, {st[2]:.4f}, {st[3]:.3f}, {st[4]:.3f}] | "
                    f"{q0:+.4f} {q1:+.4f}"
                )
            print("")
            



            if t % client_val_every == 0:
                # valida TODOS os clientes no holdout limpo (20%)
                losses = []
                accs = []
                for cid in range(n_clients):
                    losses.append(eval_loss(model_vdn, client_val_loaders[cid], max_batches=client_val_max_batches))
                    accs.append(eval_acc(model_vdn,  client_val_loaders[cid], max_batches=client_val_max_batches))

                print(
                    f"[CLIENT VAL CLEAN @ {t}] "
                    f"mean_loss={float(np.mean(losses)):.4f} | "
                    f"mean_acc={float(np.mean(accs))*100:.2f}%"
                    )




            # ===== NOVO: a cada N rounds, imprime lista de (adv=Q1-Q0) e FO (por cliente) =====
            if print_advfo_every is not None and int(print_advfo_every) > 0 and (t % int(print_advfo_every) == 0):
                adv = (q_all[:, 1] - q_all[:, 0]).astype(np.float32)
                order = np.argsort(-adv)  # desc
                print(f"[ADV/FO @ round {t}] lista (ordenada por adv=Q1-Q0 desc): cid | flag | adv | FO")
                for cid in order.tolist():
                    flag = "ATTACKER" if cid in attacked_set else "HONEST"
                    print(f"  {cid:02d} | {flag:8s} | adv={adv[cid]:+.6f} | FO={float(fo_v[cid]):+.6f}")
                print("")

            apply_fedavg(model_vdn, deltas_v, sel_v)
            update_staleness_streak(staleness_v, streak_v, sel_v)

            if (t % client_val_every) == 0:
                print(f"\n[CLIENT VAL CLEAN @ round {t}] (20% holdout, sem flip)")

                accs = np.zeros(n_clients, dtype=np.float32)
                losses = np.zeros(n_clients, dtype=np.float32)

                for cid in range(n_clients):
                    losses[cid] = float(eval_loss(model_vdn, client_val_loaders[cid], max_batches=client_val_max_batches                    ))
                    accs[cid]   = float(eval_acc(model_vdn,  client_val_loaders[cid], max_batches=client_val_max_batches                    ))

                    flag = "ATTACKER" if cid in attacked_set else "HONEST"
                    print(f"  cid {cid:02d} | {flag:8s} | loss={losses[cid]:.4f} | acc={accs[cid]*100:.2f}%")

                print(f"[CLIENT VAL CLEAN @ round {t}] mean_loss={losses.mean():.4f} | mean_acc={accs.mean()*100:.2f}%\n                ")

            
        













            l_after = eval_loss(model_vdn, val_loader, max_batches=eval_max_batches)
            loss_hist_v.append(l_after)
            r_v = windowed_reward(loss_hist_v[:-1], l_after, W=reward_window_W)
            pending_v = (obs_v.copy(), act_v.copy(), float(r_v))

            trained = False
            can_train = (
                (t >= start_train_round)
                and (t % train_every == 0)
                and (agent_v.buf.n >= batch_base)
            )
            if can_train and not force_rand:
                bs = dynamic_batch_size(agent_v.buf.n, base=batch_base, max_bs=batch_max, ratio=batch_buffer_ratio)
                _ = agent_v.train(batch_size=bs, train_steps=updates_per_round)
                trained = True

            bump_counts("vdn", sel_v)
            log["tracks"]["vdn"]["test_acc"].append(float(acc_v))

            if t % print_every == 0:
                a_r = log["tracks"]["random"]["test_acc"][-1]
                a_v = log["tracks"]["vdn"]["test_acc"][-1]
                msg = (
                    f"[summary @ {t:3d}] "
                    f"RANDOM acc={a_r*100:.2f}% | "
                    f"VDN acc={a_v*100:.2f}% | "
                    f"attacked_total={len(attacked_set)} | "
                    f"buf_n={agent_v.buf.n} | trained={int(trained)}"
                )
                print(msg, flush=True)

        # fecha transição pendente
        if pending_v is not None:
            o_prev, a_prev, r_prev = pending_v
            agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=o_prev, done=True)

        print("\nDone.")

    finally:
        save_json_safely()


if __name__ == "__main__":
    run_experiment(
        rounds=500,
        n_clients=50,
        k_select=15,
        dir_alpha=0.3,

        initial_flip_fraction=0,
        flip_add_fraction=0.0,
        attack_rounds=[600],
        flip_rate_initial=1,
        flip_rate_new_attack=0.0,

        # TARGETED:
        targeted_only_map_classes=True,
        target_map=None,

        max_per_client=None,
        local_lr=0.01,
        local_steps=10,
        probe_batches=5,

        # GLOBAL MOMENTUM
        mom_beta=0.90,

        reward_window_W=5,

        marl_eps=0.15,
        marl_swap_m=2,
        marl_lr=1e-3,
        marl_gamma=0.90,
        marl_hidden=128,
        marl_target_sync_every=20,

        warmup_transitions=50,
        start_train_round=50,
        updates_per_round=50,
        train_every=1,

        buf_size=20000,
        batch_base=32,
        batch_max=256,
        batch_buffer_ratio=4,

        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=4000,
        per_eps=1e-3,

        val_shuffle=False,
        val_per_class=200,
        eval_max_batches=20,
        print_every=1,

        # NOVO: print adv/FO
        print_advfo_every=20,

        out_dir=".",
    )
