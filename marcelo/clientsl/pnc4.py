# fl_vdn_cifar10_pool50_k15_RANDOM_vs_VDN_TARGETED_PROJ_MOM_FO_PRINT_IMIT_RANKING.py
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
#
# ADIÇÃO (SEM MEXER NO QUE JÁ EXISTIA):
# - IMITATION LEARNING (RANKING / PAIRWISE) opcional para evitar cold-start:
#   (1) Coleta dataset de imitação durante um treino (obs + máscara top-K do "professor")
#   (2) Pré-treino offline da Q-network para ordenar clientes (score = adv = Q1-Q0)
#       usando loss de ranking (BPR / logistic pairwise):
#           L = -log sigmoid( adv[pos] - adv[neg] )
#
# Como usar:
# - Para COLETAR dataset (sem pré-treino):
#     run_experiment(..., imitation_collect=True, imitation_collect_start_round=300, imitation_collect_end_round=500)
#   => salva um .npz em out_dir (por padrão, ".")
#
# - Para PRÉ-TREINAR a rede no começo de um novo run:
#     run_experiment(..., imitation_pretrain_path="imitation_ranking_seed2048_xxxxx.npz")
#
# Observação: nada muda se você deixar imitation_collect=False e imitation_pretrain_path=None.
# =================================================================================

import copy
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def log_step(msg: str):
    print(msg, flush=True)

SEED = 1000
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

class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
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

def server_reference_grad(model: nn.Module, val_loader: DataLoader, batches: int = 10) -> torch.Tensor:
    model.eval()
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

def grad_on_loader(model: nn.Module, loader: DataLoader, batches: int = 10) -> torch.Tensor:
    model.eval()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    for b, (x, y) in enumerate(loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    g = flatten_grads(model).detach()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return g






def local_train_delta(global_model: nn.Module, train_loader: DataLoader, lr: float = 0.01, steps: int = 10) -> torch.Tensor:
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
    return (w1 - w0).detach()

class SwitchableTargetedLabelFlipSubset(Dataset):
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

        if target_map is None:
            target_map = {0: 8, 8: 0, 1: 9, 9: 1, 3: 5, 5: 3, 4: 7, 7: 4, 2: 6, 6: 2}
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

def make_clients_dirichlet_indices(train_ds, n_clients: int = 50, alpha: float = 0.3, seed: int = 123, n_classes: int = 10) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(train_ds)):
        _, y = train_ds[idx]
        label_to_idxs[int(y)].append(idx)
    for y in range(n_classes):
        rng.shuffle(label_to_idxs[y])
    clients = [[] for _ in range(n_clients)]
    for y in range(n_classes):
        idxs = label_to_idxs[y]
        props = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (props * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        if diff > 0:
            for j in rng.choice(n_clients, size=diff, replace=True):
                counts[j] += 1
        elif diff < 0:
            for j in rng.choice(np.where(counts > 0)[0], size=-diff, replace=True):
                counts[j] -= 1
        start = 0
        for cid in range(n_clients):
            c = counts[cid]
            if c > 0:
                clients[cid].extend(idxs[start:start + c])
                start += c
    for cid in range(n_clients):
        rng.shuffle(clients[cid])
    return clients

def update_staleness_streak(staleness: np.ndarray, streak: np.ndarray, selected: List[int]):
    n = len(staleness)
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True
    staleness[~sel_mask] += 1.0
    staleness[sel_mask] = 0.0
    streak[~sel_mask] = 0
    streak[sel_mask] += 1

#def compute_deltas_proj_mom_probe_now_and_fo(
 #   model: nn.Module,
  #  client_train_loaders: List[DataLoader],
   # client_eval_loaders: List[DataLoader],
    #val_loader: DataLoader,
    #local_lr: float,
    #local_steps: int,
   # probe_batches: int = 1,
   # mom: Optional[torch.Tensor] = None,
   # mom_beta: float = 0.90,
#):
 #   gref = server_reference_grad(model, val_loader, batches=10)
  #  if mom is None:
   #     mom = gref.detach().clone()
    #else:
    #    mom = (mom_beta * mom) + ((1.0 - mom_beta) * gref.detach())
    #desc_mom = (-mom).detach()
    #desc_mom_norm = desc_mom / (desc_mom.norm() + 1e-12)
    #desc_gref = (-gref).detach()
    #desc_gref_norm = desc_gref / (desc_gref.norm() + 1e-12)

    #deltas, probe_now, proj_mom, fo = [], [], [], []
    #for tr_loader, ev_loader in zip(client_train_loaders, client_eval_loaders):
     #   probe_now.append(float(probing_loss(model, ev_loader, batches=probe_batches)))
      #  dw = local_train_delta(model, tr_loader, lr=local_lr, steps=local_steps)
       # deltas.append(dw)
        #proj_mom.append(float(torch.dot(dw, desc_mom_norm).item()))
        #fo.append(float(torch.dot(dw, desc_gref_norm).item()))
   # return (
    #    deltas,
    #    np.array(proj_mom, dtype=np.float32),
    #    np.array(probe_now, dtype=np.float32),
    #    np.array(fo, dtype=np.float32),
    #    mom.detach(),
    #)
#def compute_deltas_gp_score_probe_now_and_fo(
 #   model: nn.Module,
 #   client_train_loaders: List[DataLoader],
 #   client_eval_loaders: List[DataLoader],
 #   client_grad_loaders: List[DataLoader],
 #   val_loader: DataLoader,
 #   local_lr: float,
 #   local_steps: int,
 #   *,
 #   g_prev: torch.Tensor,          # OBRIGATÓRIO: ∇F(w^{t-1})
 #   probe_batches: int = 5,
 #   gref_batches: int = 10,        # só pro FO (log)
 #   client_grad_batches: int = 2,  # batches pra g_i
#) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
 #   """
 #   Retorna:
 #     deltas: Δw_i (treino local curto)
 #     gp_score: c_i^t = (g_i · g_prev)/||g_prev||   [EXATO da figura]
 #     probe_now: probing loss atual por cliente
 #     fo: <dw_i, -gref/||gref||> (log)
 #   """
 #   assert g_prev is not None, "g_prev deve ser inicializado fora do loop e passado aqui."
#
 #   g_prev = g_prev.detach()
  #  denom = float(g_prev.norm().item()) + 1e-12
#
 #   # gref atual (só para FO logging)
  #  gref = grad_on_loader(model, val_loader, batches=gref_batches).detach()
   # desc_gref_norm = (-gref) / (gref.norm() + 1e-12)
#
 #   deltas: List[torch.Tensor] = []
  #  probe_now: List[float] = []
   # gp_score: List[float] = []
    #fo: List[float] = []

    #for tr_loader, ev_loader, gr_loader in zip(client_train_loaders, client_eval_loaders,client_grad_loaders):
     #   # (1) probe loss (igual antes)
     #   probe_now.append(float(probing_loss(model, ev_loader, batches=probe_batches)))
#
 #       # (2) Δw_i via treino local (igual antes)
  #      dw = local_train_delta(model, tr_loader, lr=local_lr, steps=local_steps)
   #     deltas.append(dw)
#
 #       # (3) g_i = ∇F(w_i^t) no modelo ATUAL (sem step)
  #      gi = grad_on_loader(model, gr_loader, batches=client_grad_batches).detach()
#
 #       # (4) SCORE EXATO DA FIGURA:
  #      #     c_i^t = (gi · g_prev) / ||g_prev||
   #     score_i = float(torch.dot(gi, g_prev).item()) / denom
    #    gp_score.append(score_i)
#
 #       # (5) FO logging (igual seu jeito antigo, mas usando gref atual)
  #      fo.append(float(torch.dot(dw, desc_gref_norm).item()))
#
 #   return (
  #      deltas,
   #     np.array(gp_score, dtype=np.float32),
    #    np.array(probe_now, dtype=np.float32),
    #    np.array(fo, dtype=np.float32),
    #)


def compute_gp_probe_fo_grad_only(
    model: nn.Module,
    client_eval_loaders: List[DataLoader],
    client_grad_loaders: List[DataLoader],
    val_loader: DataLoader,
    *,
    g_prev: torch.Tensor,          # ∇F(w^{t-1})
    probe_batches: int = 5,
    gref_batches: int = 10,        # para FO_grad
    client_grad_batches: int = 2,  # batches para g_i
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sem treino local (SEM dw).
    Retorna:
      gp_score[i]  = (g_i · g_prev) / ||g_prev||      (exato da figura)
      probe_now[i] = probing loss
      fo_grad[i]   = <g_i, -gref/||gref||>            (LOG barato; opcional)
    """
    assert g_prev is not None, "g_prev deve ser passado (∇F(w^{t-1}))."
    g_prev = g_prev.detach()
    denom = float(g_prev.norm().item()) + 1e-12

    # gref atual (apenas para LOG barato)
    gref = grad_on_loader(model, val_loader, batches=gref_batches).detach()
    desc_gref_norm = (-gref) / (gref.norm() + 1e-12)

    gp_score: List[float] = []
    probe_now: List[float] = []
    fo_grad: List[float] = []

    for ev_loader, gr_loader in zip(client_eval_loaders, client_grad_loaders):
        probe_now.append(float(probing_loss(model, ev_loader, batches=probe_batches)))

        gi = grad_on_loader(model, gr_loader, batches=client_grad_batches).detach()

        score_i = float(torch.dot(gi, g_prev).item()) / denom
        gp_score.append(score_i)

        fo_grad.append(float(torch.dot(gi, desc_gref_norm).item()))

    return (
        np.array(gp_score, dtype=np.float32),
        np.array(probe_now, dtype=np.float32),
        np.array(fo_grad, dtype=np.float32),
    )


def apply_fedavg_selected(model: nn.Module, delta_by_cid: Dict[int, torch.Tensor], selected: List[int]):
    """Aplica FedAvg apenas com os deltas dos clientes selecionados."""
    w = flatten_params(model).clone()
    avg_dw = torch.stack([delta_by_cid[cid] for cid in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)





def local_train_model_and_delta(
    global_model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.01,
    steps: int = 10,
) -> Tuple[nn.Module, torch.Tensor]:
    m = copy.deepcopy(global_model).to(DEVICE)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)

    w0 = flatten_params(m).clone()

    it = iter(train_loader)
    for _ in range(int(steps)):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()

    w1 = flatten_params(m).detach()
    dw = (w1 - w0).detach()
    return m, dw    


def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor], selected: List[int]):
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)

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

def build_context_matrix_vdn(projection_mom: np.ndarray, probe_now: np.ndarray, staleness: np.ndarray, streak: np.ndarray) -> np.ndarray:
    proj = projection_mom.astype(np.float32)
    probe = probe_now.astype(np.float32)
    s = staleness.astype(np.float32)
    sn = (s / float(s.max() + 1e-6)).astype(np.float32)
    cap = 5.0
    tn = np.clip(streak.astype(np.float32) / cap, 0.0, 1.0).astype(np.float32)
    bias = np.ones((proj.shape[0],), dtype=np.float32)
    return np.stack([bias, proj, probe, sn, tn], axis=1).astype(np.float32)

class PrioritizedReplayJoint:
    def __init__(self, capacity: int, n_agents: int, d_in: int, alpha: float = 0.6, eps: float = 1e-3, seed: int = 0):
        self.capacity = int(capacity); self.n_agents = int(n_agents); self.d_in = int(d_in)
        self.alpha = float(alpha); self.eps = float(eps)
        self.obs  = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.act  = np.zeros((capacity, n_agents), dtype=np.uint8)
        self.r    = np.zeros((capacity,), dtype=np.float32)
        self.obs2 = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.p = np.zeros((capacity,), dtype=np.float32)
        self.n = 0; self.ptr = 0; self.max_p = 1.0
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
        pri = self.p[:self.n].astype(np.float64)
        probs = (pri + self.eps) ** self.alpha
        s = probs.sum()
        probs = (np.ones_like(probs) / len(probs)) if s <= 0 else (probs / s)
        idx = self.rng.choice(self.n, size=bs, replace=False, p=probs)
        w = (self.n * probs[idx]) ** (-beta)
        w = w / (w.max() + 1e-12)
        return (self.obs[idx], self.act[idx], self.r[idx], self.obs2[idx], self.done[idx], idx.astype(np.int64), w.astype(np.float32))

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.p[idx] = td_abs + self.eps
        self.max_p = float(max(self.max_p, float(td_abs.max(initial=0.0))))

class AgentMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class VDNSelector:
    def __init__(self, n_agents: int, d_in: int, k_select: int, hidden: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4,
                 gamma: float = 0.90, grad_clip: float = 1.0, target_sync_every: int = 20, buf_size: int = 20000,
                 batch_size: int = 128, train_steps: int = 20, per_alpha: float = 0.6, per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0, per_beta_steps: int = 4000, per_eps: float = 1e-3, double_dqn: bool = True, seed: int = 0):
        self.n_agents = int(n_agents); self.d_in = int(d_in); self.k_select = int(k_select)
        self.gamma = float(gamma); self.grad_clip = float(grad_clip); self.target_sync_every = int(target_sync_every); self.double_dqn = bool(double_dqn)
        self.batch_size = int(batch_size); self.train_steps = int(train_steps)
        self.per_beta_start = float(per_beta_start); self.per_beta_end = float(per_beta_end); self.per_beta_steps = int(per_beta_steps)
        self.q = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.buf = PrioritizedReplayJoint(capacity=int(buf_size), n_agents=self.n_agents, d_in=self.d_in, alpha=float(per_alpha), eps=float(per_eps), seed=int(seed) + 12345)
        self._train_calls = 0
        self.per_eps = float(per_eps)
        self.py_rng = random.Random(int(seed) + 777)
        self.np_rng = np.random.default_rng(int(seed) + 999)

    def _beta(self) -> float:
        t = min(self._train_calls, self.per_beta_steps)
        frac = t / max(1, self.per_beta_steps)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    @torch.no_grad()
    def _q_all_agents(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        q = self.q(x)
        return q.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        q = self.q(x)
        return q.detach().cpu().numpy().astype(np.float32)

    def select_topk_actions(self, obs: np.ndarray, eps: float = 0.15, swap_m: int = 2, force_random: bool = False):
        n = obs.shape[0]; K = min(self.k_select, n)
        q = self._q_all_agents(obs)
        if force_random:
            sel = self.py_rng.sample(range(n), K)
            a = np.zeros(n, dtype=np.uint8); a[sel] = 1
            return a, sel
        adv = q[:, 1] - q[:, 0]
        sel = np.argsort(adv)[::-1][:K].tolist()
        if swap_m > 0 and (self.np_rng.random() < eps):
            swap_m = min(swap_m, K)
            sel_set = set(sel); not_sel = [i for i in range(n) if i not in sel_set]
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
        a = np.zeros(n, dtype=np.uint8); a[sel] = 1
        return a, sel

    def add_transition(self, obs, act, r, obs2, done: bool):
        self.buf.add(obs=obs, act=act, r=r, obs2=obs2, done=done)

    def train(self, batch_size: Optional[int] = None, train_steps: Optional[int] = None):
        bs_req = int(batch_size) if batch_size is not None else self.batch_size
        steps = int(train_steps) if train_steps is not None else self.train_steps
        if self.buf.n < max(32, bs_req):
            return None
        beta = self._beta()
        self.q.train()
        losses = []
        for _ in range(steps):
            ob, ac, rw, ob2, dn, idx, w_is = self.buf.sample(batch_size=bs_req, beta=beta)
            B, N, D = ob.shape
            obs  = torch.tensor(ob,  dtype=torch.float32, device=DEVICE)
            act  = torch.tensor(ac,  dtype=torch.long,   device=DEVICE)
            r    = torch.tensor(rw,  dtype=torch.float32, device=DEVICE)
            obs2 = torch.tensor(ob2, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(dn,  dtype=torch.float32, device=DEVICE)
            w    = torch.tensor(w_is, dtype=torch.float32, device=DEVICE)
            q_cur = self.q(obs.reshape(B * N, D)).reshape(B, N, 2)
            q_a = q_cur.gather(2, act.unsqueeze(2)).squeeze(2)
            q_tot = q_a.sum(dim=1)
            with torch.no_grad():
                q2_online = self.q(obs2.reshape(B * N, D)).reshape(B, N, 2)
                q2_tgt    = self.q_tgt(obs2.reshape(B * N, D)).reshape(B, N, 2)
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
            self.opt.zero_grad(); loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()
            self.buf.update_priorities(idx, td_abs)
            losses.append(float(loss.item()))
        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
        return float(np.mean(losses)) if losses else None

    # ---- IMITATION PRETRAIN (RANKING) ----
    def imitation_pretrain_ranking(self, npz_path: str, epochs: int = 5, batch_transitions: int = 256, pairs_per_transition: int = 64,
                                   lr: float = 1e-3, weight_decay: float = 1e-4, grad_clip: float = 1.0, temperature: float = 1.0,
                                   seed: int = 0, verbose: bool = True):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"imitation_pretrain_ranking: arquivo não existe: {npz_path}")
        data = np.load(str(path), allow_pickle=False)
        obs = data["obs"].astype(np.float32)
        sel = data["sel_mask"].astype(np.uint8)
        T, N, D = obs.shape
        if N != self.n_agents or D != self.d_in:
            raise ValueError(f"Dataset mismatch: obs=(T={T},N={N},D={D}) vs selector (N={self.n_agents},D={self.d_in})")
        rng = np.random.default_rng(int(seed) + 123)
        opt = torch.optim.Adam(self.q.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        idx_all = np.arange(T)
        self.q.train()
        for ep in range(1, int(epochs) + 1):
            rng.shuffle(idx_all)
            losses = []
            for start in range(0, T, int(batch_transitions)):
                bidx = idx_all[start:start + int(batch_transitions)]
                ob = obs[bidx]; sm = sel[bidx]
                B = ob.shape[0]
                x = torch.tensor(ob, dtype=torch.float32, device=DEVICE).reshape(B * N, D)
                q = self.q(x).reshape(B, N, 2)
                adv = (q[:, :, 1] - q[:, :, 0])
                terms = []
                for b in range(B):
                    pos = np.where(sm[b] == 1)[0]
                    neg = np.where(sm[b] == 0)[0]
                    if pos.size == 0 or neg.size == 0:
                        continue
                    P = int(pairs_per_transition)
                    pos_s = rng.choice(pos, size=P, replace=True)
                    neg_s = rng.choice(neg, size=P, replace=True)
                    s_pos = adv[b, torch.tensor(pos_s, device=DEVICE, dtype=torch.long)]
                    s_neg = adv[b, torch.tensor(neg_s, device=DEVICE, dtype=torch.long)]
                    diff = (s_pos - s_neg) / max(1e-6, float(temperature))
                    terms.append(F.softplus(-diff).mean())
                if not terms:
                    continue
                loss = torch.stack(terms).mean()
                opt.zero_grad(); loss.backward()
                if grad_clip and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(self.q.parameters(), float(grad_clip))
                opt.step()
                losses.append(float(loss.item()))
            self.q_tgt.load_state_dict(self.q.state_dict())
            if verbose:
                m = float(np.mean(losses)) if losses else float("nan")
                print(f"[imitation-pretrain][ranking] epoch {ep}/{epochs} | loss={m:.6f} | T={T} | pairs/transition={pairs_per_transition}", flush=True)

class ImitationRankingCollector:
    def __init__(self, n_agents: int, d_in: int, out_path: Path, meta: dict):
        self.n_agents = int(n_agents); self.d_in = int(d_in)
        self.out_path = Path(out_path); self.meta = dict(meta)
        self._obs = []; self._sel = []; self._rounds = []

    def add(self, round_t: int, obs: np.ndarray, selected: List[int]):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.n_agents, self.d_in):
            raise ValueError(f"Collector.add: obs shape {obs.shape} != {(self.n_agents, self.d_in)}")
        sel_mask = np.zeros((self.n_agents,), dtype=np.uint8)
        sel_mask[np.asarray(selected, dtype=np.int64)] = 1
        self._obs.append(obs); self._sel.append(sel_mask); self._rounds.append(int(round_t))

    def save(self):
        if len(self._obs) == 0:
            print("[imitation-collect] nenhum dado coletado, nada para salvar.", flush=True)
            return None
        obs = np.stack(self._obs, axis=0).astype(np.float32)
        sel = np.stack(self._sel, axis=0).astype(np.uint8)
        rounds = np.array(self._rounds, dtype=np.int32)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.out_path), obs=obs, sel_mask=sel, rounds=rounds, meta_json=json.dumps(self.meta))
        print(f"\\n[IMITATION DATASET] salvo em: {str(self.out_path)} | T={obs.shape[0]} transições\\n", flush=True)
        return str(self.out_path)

# --- run_experiment + main ---
def run_experiment(rounds: int = 300, n_clients: int = 50, k_select: int = 15, dir_alpha: float = 0.3,
    initial_flip_fraction: float = 0.0, flip_add_fraction: float = 0.20, attack_rounds: List[int] = None,
    flip_rate_initial: float = 1.0, flip_rate_new_attack: float = 1.0, targeted_only_map_classes: bool = True,
    target_map: Optional[Dict[int, int]] = None, max_per_client: int = 2500, local_lr: float = 0.01, local_steps: int = 10,
    probe_batches: int = 5, mom_beta: float = 0.90, reward_window_W: int = 5, marl_eps: float = 0.15, marl_swap_m: int = 2,
    marl_lr: float = 1e-3, marl_gamma: float = 0.90, marl_hidden: int = 128, marl_target_sync_every: int = 20,
    warmup_transitions: int = 200, start_train_round: int = 100, updates_per_round: int = 50, train_every: int = 1,
    buf_size: int = 20000, batch_base: int = 64, batch_max: int = 256, batch_buffer_ratio: int = 4, per_alpha: float = 0.6,
    per_beta_start: float = 0.4, per_beta_end: float = 1.0, per_beta_steps: int = 4000, per_eps: float = 1e-3,
    val_shuffle: bool = False, val_per_class: int = 200, eval_max_batches: int = 20, print_every: int = 10,
    print_advfo_every: int = 20, out_dir: str = ".",
    imitation_collect: bool = False, imitation_collect_start_round: int = 300, imitation_collect_end_round: Optional[int] = None,
    imitation_collect_every: int = 1, imitation_collect_skip_if_forced_random: bool = True, imitation_collect_skip_before_train_round: bool = True,
    imitation_pretrain_path: Optional[str] = None, imitation_pretrain_epochs: int = 5, imitation_pretrain_batch_transitions: int = 256,
    imitation_pretrain_pairs_per_transition: int = 64, imitation_pretrain_lr: float = 1e-3, imitation_pretrain_weight_decay: float = 1e-4,
    imitation_pretrain_grad_clip: float = 1.0, imitation_pretrain_temperature: float = 1.0):

    if attack_rounds is None:
        attack_rounds = [150]
    attack_rounds = sorted(list(set(int(x) for x in attack_rounds)))
    mean = (0.4914, 0.4822, 0.4465); std  = (0.2470, 0.2435, 0.2616)

    tfm_train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
                                       )
    tfm_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
])

    train_ds_aug  = datasets.CIFAR10(root="./data", train=True,  download=True,  transform=tfm_train_aug)
    train_ds_eval = datasets.CIFAR10(root="./data", train=True,  download=False, transform=tfm_eval)
    test_ds       = datasets.CIFAR10(root="./data", train=False, download=True,  transform=tfm_eval)
    



    #tfm_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    #tfm_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    run_id = uuid.uuid4().hex[:10]
    out_path = Path(out_dir) / f"results_random_vs_vdn_targeted_PROJMOM_FO_seed{SEED}_{run_id}.json"
    imitation_out_path = Path(out_dir) / f"imitation_ranking_seed{SEED}_{run_id}.npz"

    log = {"meta": {"run_id": run_id, "seed": int(SEED), "device": str(DEVICE), "rounds": int(rounds), "n_clients": int(n_clients), "k_select": int(k_select),
                    "dir_alpha": float(dir_alpha), "initial_flip_fraction": float(initial_flip_fraction), "flip_add_fraction": float(flip_add_fraction),
                    "attack_rounds": list(attack_rounds), "flip_rate_initial": float(flip_rate_initial), "flip_rate_new_attack": float(flip_rate_new_attack),
                    "targeted_only_map_classes": bool(targeted_only_map_classes), "target_map": (target_map if target_map is not None else "default_pair_swaps"),
                    "mom_beta": float(mom_beta), "buf_size": int(buf_size), "warmup_transitions": int(warmup_transitions), "start_train_round": int(start_train_round),
                    "updates_per_round": int(updates_per_round), "train_every": int(train_every), "print_advfo_every": int(print_advfo_every),
                    "imitation_collect": bool(imitation_collect), "imitation_collect_start_round": int(imitation_collect_start_round),
                    "imitation_collect_end_round": (None if imitation_collect_end_round is None else int(imitation_collect_end_round)),
                    "imitation_collect_every": int(imitation_collect_every), "imitation_collect_skip_if_forced_random": bool(imitation_collect_skip_if_forced_random),
                    "imitation_collect_skip_before_train_round": bool(imitation_collect_skip_before_train_round),
                    "imitation_pretrain_path": (None if imitation_pretrain_path is None else str(imitation_pretrain_path)),
                    "imitation_pretrain_epochs": int(imitation_pretrain_epochs), "imitation_pretrain_batch_transitions": int(imitation_pretrain_batch_transitions),
                    "imitation_pretrain_pairs_per_transition": int(imitation_pretrain_pairs_per_transition), "imitation_pretrain_temperature": float(imitation_pretrain_temperature)},
           "attack_schedule": [], "tracks": {"random": {"test_acc": [], "selection_count_total_per_client": [0] * int(n_clients), "selection_phases": []},
                                            "vdn":    {"test_acc": [], "selection_count_total_per_client": [0] * int(n_clients), "selection_phases": []}}}

    def save_json_safely():
        for key in ["random", "vdn"]:
            cnt_total = np.array(log["tracks"][key]["selection_count_total_per_client"], dtype=np.int64)
            log["tracks"][key]["final_metrics"] = {"gini_selection_total": float(gini_coefficient(cnt_total)), "total_selections": int(cnt_total.sum())}
            phases = log["tracks"][key]["selection_phases"]
            for ph in phases:
                if ph.get("end_round", None) is None:
                    ph["end_round"] = int(rounds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"\\n[JSON] salvo em: {str(out_path)}\\n", flush=True)

    def start_new_phase(track_key: str, start_round: int, attacked_snapshot: List[int]):
        log["tracks"][track_key]["selection_phases"].append({"start_round": int(start_round), "end_round": None, "attacked_clients_snapshot": list(attacked_snapshot),
                                                            "selection_count_per_client": [0] * int(n_clients)})

    def bump_counts(track_key: str, selected: List[int]):
        total_cnt = log["tracks"][track_key]["selection_count_total_per_client"]
        phase_cnt = log["tracks"][track_key]["selection_phases"][-1]["selection_count_per_client"]
        for i in selected:
            total_cnt[i] += 1
            phase_cnt[i] += 1

    log_step("Baixando/carregando CIFAR-10...")
    #train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    #test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    log_step("Criando server_val balanceado (holdout do TRAIN) + train_pool (sem val)...")
    #server_val_idxs = make_server_val_balanced(train_ds, per_class=val_per_class, n_classes=10, seed=SEED + 4242)
    server_val_idxs = make_server_val_balanced(train_ds_eval, per_class=val_per_class, n_classes=10, seed=SEED + 4242)
    server_val_set = set(server_val_idxs)

    server_val_set = set(server_val_idxs)
    all_train_idxs = np.arange(len(train_ds_eval))
    train_pool_idxs = [int(i) for i in all_train_idxs if int(i) not in server_val_set]
    #train_pool = Subset(train_ds, train_pool_idxs)

    train_pool_aug  = Subset(train_ds_aug,  train_pool_idxs)   # treino local
    train_pool_eval = Subset(train_ds_eval, train_pool_idxs)   # probing/grad

    g_val = torch.Generator(); g_val.manual_seed(SEED + 123)
    #val_loader = DataLoader(Subset(train_ds, server_val_idxs), batch_size=256, shuffle=val_shuffle, generator=g_val, worker_init_fn=seed_worker, num_workers=0)
    val_loader = DataLoader(
        Subset(train_ds_eval, server_val_idxs),
        batch_size=256,
        shuffle=val_shuffle,
        generator=g_val,
        worker_init_fn=seed_worker,
        num_workers=0
        )

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    log_step(f"Gerando split Dirichlet (alpha={dir_alpha}) para {n_clients} clientes (em train_pool)...")
    #client_idxs = make_clients_dirichlet_indices(train_pool, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777, n_classes=10)
    client_idxs = make_clients_dirichlet_indices(train_pool_eval, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777,n_classes=10)
    n_init = int(round(initial_flip_fraction * n_clients))
    rng_init = np.random.RandomState(SEED + 999)
    attacked_set = set(rng_init.choice(np.arange(n_clients), size=n_init, replace=False).tolist()) if n_init > 0 else set()
    attack_rate_per_client = np.zeros(n_clients, dtype=np.float32)
    for cid in attacked_set:
        attack_rate_per_client[cid] = float(flip_rate_initial)

    client_train_loaders, client_eval_loaders, client_grad_loaders = [], [], []
    client_sizes = []
    switchable_ds = [] #guarda os pares train,eval
    g_train = torch.Generator(); g_train.manual_seed(SEED + 10001)

    log_step("Criando loaders dos clientes + label flipping TARGETED determinístico (switchable, taxa ajustável)...")
    for cid, idxs in enumerate(client_idxs):
        if max_per_client is not None:
            idxs = idxs[:max_per_client]
        #client_sizes.append(len(idxs))
        #ds_c = SwitchableTargetedLabelFlipSubset(train_pool, idxs, n_classes=10, seed=SEED + 1000 + cid, enabled=(cid in attacked_set),attack_rate=float(attack_rate_per_client[cid]), target_map=target_map, only_map_classes=targeted_only_map_classes)
        # --- dataset para TREINO (com AUG) --
        ds_train = SwitchableTargetedLabelFlipSubset(
            train_pool_aug, idxs,
            n_classes=10, seed=SEED + 1000 + cid,
            enabled=(cid in attacked_set),
            attack_rate=float(attack_rate_per_client[cid]),
            target_map=target_map,
            only_map_classes=targeted_only_map_classes
            )   

        # --- dataset para EVAL/GRAD (sem AUG) ---
        ds_eval = SwitchableTargetedLabelFlipSubset(
            train_pool_eval, idxs,
            n_classes=10, seed=SEED + 1000 + cid,
            enabled=(cid in attacked_set),
            attack_rate=float(attack_rate_per_client[cid]),
            target_map=target_map,
            only_map_classes=targeted_only_map_classes
    )


        #switchable_ds.append(ds_c)
        switchable_ds.append((ds_train, ds_eval))

        #client_train_loaders.append(DataLoader(ds_c, batch_size=64, shuffle=True, generator=g_train, worker_init_fn=seed_worker, num_workers=0))
        #client_eval_loaders.append(DataLoader(ds_c, batch_size=64, shuffle=False, num_workers=0))


        client_train_loaders.append(DataLoader(
            ds_train, batch_size=64, shuffle=True,
            generator=g_train, worker_init_fn=seed_worker, num_workers=0
        ))
    

        # probing loss (sem AUG, determinístico)
        client_eval_loaders.append(DataLoader(
            ds_eval, batch_size=64, shuffle=False, num_workers=0
            ))


        # grad do cliente g_i (sem AUG) 
        client_grad_loaders.append(DataLoader(
        ds_eval, batch_size=128, shuffle=False,
        generator=g_train, worker_init_fn=seed_worker, num_workers=0
    ))
    

    base = SmallCNN().to(DEVICE)

    model_rand = copy.deepcopy(base).to(DEVICE)
    rng_random_sel = random.Random(SEED + 424242)
    start_new_phase("random", 1, sorted(list(attacked_set)))

    model_vdn = copy.deepcopy(base).to(DEVICE)
    staleness_v = np.zeros(n_clients, dtype=np.float32)
    streak_v = np.zeros(n_clients, dtype=np.int32)
    loss_hist_v = []
    pending_v = None
    mom_v = None

    agent_v = VDNSelector(n_agents=n_clients, d_in=5, k_select=k_select, hidden=marl_hidden, lr=marl_lr, weight_decay=1e-4, gamma=marl_gamma,
                          grad_clip=1.0, target_sync_every=marl_target_sync_every, buf_size=buf_size, batch_size=batch_base, train_steps=max(1, updates_per_round),
                          per_alpha=per_alpha, per_beta_start=per_beta_start, per_beta_end=per_beta_end, per_beta_steps=per_beta_steps, per_eps=per_eps, double_dqn=True, seed=SEED + 10)

    if imitation_pretrain_path is not None:
        print(f"\\n[IMITATION PRETRAIN] carregando dataset: {imitation_pretrain_path}\\n", flush=True)
        agent_v.imitation_pretrain_ranking(npz_path=str(imitation_pretrain_path), epochs=imitation_pretrain_epochs, batch_transitions=imitation_pretrain_batch_transitions,
                                           pairs_per_transition=imitation_pretrain_pairs_per_transition, lr=imitation_pretrain_lr, weight_decay=imitation_pretrain_weight_decay,
                                           grad_clip=imitation_pretrain_grad_clip, temperature=imitation_pretrain_temperature, seed=SEED + 2025, verbose=True)
        print("[IMITATION PRETRAIN] concluído.\\n", flush=True)

    start_new_phase("vdn", 1, sorted(list(attacked_set)))

    collector = None
    if imitation_collect:
        meta = {"seed": int(SEED), "run_id": run_id, "n_clients": int(n_clients), "k_select": int(k_select), "d_in": 5,
                "state_desc": "[bias, proj_mom, probe_now, staleness_n, streak_n]", "collect_start_round": int(imitation_collect_start_round),
                "collect_end_round": (None if imitation_collect_end_round is None else int(imitation_collect_end_round)), "collect_every": int(imitation_collect_every),
                "skip_if_forced_random": bool(imitation_collect_skip_if_forced_random), "skip_before_train_round": bool(imitation_collect_skip_before_train_round)}
        collector = ImitationRankingCollector(n_agents=n_clients, d_in=5, out_path=imitation_out_path, meta=meta)
        print(f"\\n[IMITATION COLLECT] ATIVO: vai salvar em {str(imitation_out_path)}\\n", flush=True)

    # ===== GP: inicializa g_prev (grad do servidor no w^0) =====
    g_prev_batches = 10  

    g_prev_rand = grad_on_loader(model_rand, val_loader, batches=g_prev_batches).detach()  # ∇F(w^0)
    g_prev_vdn  = grad_on_loader(model_vdn,  val_loader, batches=g_prev_batches).detach()  # ∇F(w^0)


    try:
        for t in range(1, int(rounds) + 1):
            log_step(f"\\n[round {t}/{rounds}] começando...")
            if t in attack_rounds:
                n_add = int(round(flip_add_fraction * n_clients))
                candidates = [i for i in range(n_clients) if i not in attacked_set]
                rng_add = np.random.RandomState(SEED + 5000 + t)
                rng_add.shuffle(candidates)
                add_now = candidates[:min(n_add, len(candidates))]
                for cid in add_now:
                    attacked_set.add(cid); attack_rate_per_client[cid] = float(flip_rate_new_attack)
                #for cid, ds in enumerate(switchable_ds):
                    #ds.set_attack(cid in attacked_set, float(attack_rate_per_client[cid]) if cid in attacked_set else 0.0)
                for cid, (ds_tr, ds_ev) in enumerate(switchable_ds):
                    rate = float(attack_rate_per_client[cid]) if cid in attacked_set else 0.0
                    ds_tr.set_attack(cid in attacked_set, rate)
                    ds_ev.set_attack(cid in attacked_set, rate) 


                log["attack_schedule"].append({"round": int(t), "added_clients": list(map(int, add_now)), "rate_for_added": float(flip_rate_new_attack),
                                              "attacked_total_after": int(len(attacked_set))})
                log_step(f"  >>> ATTACK ADD @ round {t}: adicionados {len(add_now)} novos atacados | attacked_total={len(attacked_set)}")

            round_seed = SEED + 50000 + t
            g_train.manual_seed(round_seed)

            a_rand = eval_acc(model_rand, test_loader, max_batches=80)


            #deltas_r, _, _, _= compute_deltas_gp_score_probe_now_and_fo(model_rand, client_train_loaders, client_eval_loaders, client_grad_loaders, val_loader, local_lr, local_steps, g_prev=g_prev_rand, probe_batches=probe_batches,gref_batches=10, client_grad_batches=2)




            K = min(k_select, n_clients)
            sel_r = rng_random_sel.sample(range(n_clients), K)

            delta_by_cid_r = {}
            for cid in sel_r:
                delta_by_cid_r[cid] = local_train_delta(model_rand, client_train_loaders[cid], lr=local_lr, steps=local_steps)



            apply_fedavg(model_rand, delta_by_cid_r, sel_r)
            bump_counts("random", sel_r)
            log["tracks"]["random"]["test_acc"].append(float(a_rand))

            
            g_prev_rand = grad_on_loader(model_rand, val_loader, batches=g_prev_batches).detach()  # agora é ∇F(w^t)

            acc_v = eval_acc(model_vdn, test_loader, max_batches=80)

            gp_v, probe_now_v, fo_grad_v = compute_gp_probe_fo_grad_only(
                model_vdn,
                client_eval_loaders,
                client_grad_loaders,
                val_loader,
                g_prev=g_prev_vdn,
                probe_batches=probe_batches,
                gref_batches=10,
                client_grad_batches=2,
                )

            #deltas_v, gp_v, probe_now_v, fo_v = compute_deltas_gp_score_probe_now_and_fo(model_vdn, client_train_loaders, client_eval_loaders, client_grad_loaders, val_loader, local_lr, local_steps, g_prev=g_prev_vdn, probe_batches=probe_batches,gref_batches=10, client_grad_batches=2)
            obs_v = build_context_matrix_vdn(gp_v, probe_now_v, staleness_v, streak_v)

            if pending_v is not None:
                o_prev, a_prev, r_prev = pending_v
                agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=obs_v, done=False)

            force_rand = (agent_v.buf.n < warmup_transitions)


            #
            swap_eps = 0.20
            half = rounds *0.6
            if t <=half:
                swap_m_sched = 5
            else:
                swap_m_sched = 2


            act_v, sel_v = agent_v.select_topk_actions(obs=obs_v, eps=marl_eps, swap_m=marl_swap_m, force_random=force_rand)

            if collector is not None and imitation_collect:
                end_r = int(rounds) if imitation_collect_end_round is None else int(imitation_collect_end_round)
                in_window = (t >= int(imitation_collect_start_round)) and (t <= end_r)
                every_ok = (int(imitation_collect_every) <= 1) or (t % int(imitation_collect_every) == 0)
                skip = False
                if imitation_collect_skip_if_forced_random and force_rand:
                    skip = True
                if imitation_collect_skip_before_train_round and (t < int(start_train_round)):
                    skip = True
                if in_window and every_ok and (not skip):
                    collector.add(round_t=t, obs=obs_v.copy(), selected=sel_v)

            q_all = agent_v.q_values(obs_v)
            print("\\n[SELECTED DEBUG] cid | flag | state=[bias, proj_mom, probe_now, stale_n, streak_n] | Q0 Q1")
            for cid in sel_v:
                flag = "ATTACKER" if cid in attacked_set else "HONEST"
                st = obs_v[cid]
                q0, q1 = float(q_all[cid, 0]), float(q_all[cid, 1])
                print(f"  {cid:02d} | {flag:8s} | [{st[0]:.3f}, {st[1]:+.4f}, {st[2]:.4f}, {st[3]:.3f}, {st[4]:.3f}] | {q0:+.4f} {q1:+.4f}")
            print("")

            if print_advfo_every is not None and int(print_advfo_every) > 0 and (t % int(print_advfo_every) == 0):
                adv = (q_all[:, 1] - q_all[:, 0]).astype(np.float32)
                order = np.argsort(-adv)
                print(f"[ADV/FO @ round {t}] lista (ordenada por adv=Q1-Q0 desc): cid | flag | adv | FO")
                for cid in order.tolist():
                    flag = "ATTACKER" if cid in attacked_set else "HONEST"
                    print(f"  {cid:02d} | {flag:8s} | adv={adv[cid]:+.6f} | FO={float(fo_v[cid]):+.6f}")
                print("")


            delta_by_cid_v = {}
            for cid in sel_v:
                delta_by_cid_v[cid] = local_train_delta(model_vdn, client_train_loaders[cid], lr=local_lr, steps=local_steps)





            apply_fedavg(model_vdn, delta_by_cid_v, sel_v)
            g_prev_vdn = grad_on_loader(model_vdn, val_loader, batches=g_prev_batches).detach()
            update_staleness_streak(staleness_v, streak_v, sel_v)
            l_after = eval_loss(model_vdn, val_loader, max_batches=eval_max_batches)
            loss_hist_v.append(l_after)
            r_v = windowed_reward(loss_hist_v[:-1], l_after, W=reward_window_W)
            pending_v = (obs_v.copy(), act_v.copy(), float(r_v))

            trained = False
            can_train = (t >= int(start_train_round)) and (t % int(train_every) == 0) and (agent_v.buf.n >= int(batch_base))
            if can_train and not force_rand:
                bs = dynamic_batch_size(agent_v.buf.n, base=batch_base, max_bs=batch_max, ratio=batch_buffer_ratio)
                _ = agent_v.train(batch_size=bs, train_steps=updates_per_round)
                trained = True

            bump_counts("vdn", sel_v)
            log["tracks"]["vdn"]["test_acc"].append(float(acc_v))

            if t % int(print_every) == 0:
                a_r = log["tracks"]["random"]["test_acc"][-1]
                a_v = log["tracks"]["vdn"]["test_acc"][-1]
                print(f"[summary @ {t:3d}] RANDOM acc={a_r*100:.2f}% | VDN acc={a_v*100:.2f}% | attacked_total={len(attacked_set)} | buf_n={agent_v.buf.n} | trained={int(trained)}", flush=True)

        if pending_v is not None:
            o_prev, a_prev, r_prev = pending_v
            agent_v.add_transition(obs=o_prev, act=a_prev, r=r_prev, obs2=o_prev, done=True)

    finally:
        save_json_safely()
        if collector is not None and imitation_collect:
            collector.save()

if __name__ == "__main__":
    run_experiment(
        rounds=500, 
        n_clients=50,
        k_select=10,
        dir_alpha=0.3,
        initial_flip_fraction=0.4,
        flip_add_fraction=0,
        attack_rounds=[600],
        flip_rate_initial=0.7,
        flip_rate_new_attack=0.0,
        targeted_only_map_classes=True, 
        target_map=None,
        max_per_client=2500, 
        local_lr=0.01,
        local_steps=10,
        probe_batches=5,
        mom_beta=0.90,
        reward_window_W=5, 
        marl_eps=0.15, 
        marl_swap_m=2, 
        marl_lr=1e-3, 
        marl_gamma=0.90, 
        marl_hidden=128, 
        marl_target_sync_every=20,
        warmup_transitions=100, 
        start_train_round=100, 
        updates_per_round=50, 
        train_every=1, 
        buf_size=20000, 
        batch_base=64, 
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
        print_advfo_every=20, 
        out_dir=".",
        imitation_collect=False, 
        imitation_collect_start_round=300, 
        imitation_collect_end_round=500, 
        imitation_collect_every=1,
        imitation_collect_skip_if_forced_random=False, 
        imitation_collect_skip_before_train_round=False,
        imitation_pretrain_path= None, 
        imitation_pretrain_epochs=150, 
        imitation_pretrain_batch_transitions=64, 
        imitation_pretrain_pairs_per_transition=64,
        imitation_pretrain_lr=1e-3, 
        imitation_pretrain_weight_decay=1e-4, 
        imitation_pretrain_grad_clip=1.0, 
        imitation_pretrain_temperature=1.0)

