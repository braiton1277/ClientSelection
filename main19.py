"""
FL client selection experiment (MNIST) — RANDOM vs DQN (1-step TD) selector
(+ optional METRIC track kept for reference)

Upgrade from contextual bandit -> RL:
- Replace "context -> reward regression" with Q-learning:
    Q(x,a) where a in {0=not select, 1=select}
- Each round produces contexts X_t for all clients.
- We select K clients (Top-K by advantage Q(x,1)-Q(x,0), with eps exploration).
- Reward per round: val_loss_before - val_loss_after (positive is good)
- Credit assignment (simple baseline): each selected client gets reward_round/K
- We also sample some unselected clients each round with a=0 and r=0 to stabilize Q(x,0).
- Transition is built one round later: (x_t, a_t, r_t, x_{t+1}, done=0)
- Target network + Double DQN (stable).

Run:
  python this_file.py
"""

import copy
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================
# Logging helper
# ============================
def log_step(msg: str):
    print(msg, flush=True)


# ============================
# Reproducibility
# ============================
SEED = 123
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


g_dl = torch.Generator()
g_dl.manual_seed(SEED)


# ============================
# Model (global model on each track)
# ============================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
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


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm() + eps)).item()


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
def probing_loss(model: nn.Module, loader: DataLoader, batches: int = 2) -> float:
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
def loss_on_few_batches(model: nn.Module, loader: DataLoader, batches: int = 2) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        tot += F.cross_entropy(model(x), y).item()
        n += 1
    return tot / max(1, n)


# ============================
# Server reference gradient
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
# Local training delta + local delta-loss
# ============================
def local_train_delta_and_dloss(
    global_model: nn.Module,
    loader: DataLoader,
    lr: float = 0.05,
    steps: int = 30,
    dloss_batches: int = 2,
) -> Tuple[torch.Tensor, float]:
    model = copy.deepcopy(global_model).to(DEVICE)

    loss_before = loss_on_few_batches(model, loader, batches=dloss_batches)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    w0 = flatten_params(model).clone()

    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    loss_after = loss_on_few_batches(model, loader, batches=dloss_batches)

    w1 = flatten_params(model)
    dw = (w1 - w0).detach()
    dloss = float(loss_before - loss_after)
    return dw, dloss


# ============================
# Label flipping wrapper
# ============================
class LabelFlipSubset(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices, flip_rate: float, n_classes: int = 10, seed: int = 0):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.flip_rate = float(flip_rate)
        self.n_classes = int(n_classes)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base_ds[self.indices[i]]
        y = int(y)
        if self.flip_rate > 0.0 and self.rng.rand() < self.flip_rate:
            y_new = self.rng.randint(0, self.n_classes - 1)
            if y_new >= y:
                y_new += 1
            y = y_new
        return x, y


# ============================
# Server balanced validation
# ============================
def make_server_val_balanced(ds, per_class: int = 200) -> List[int]:
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        label_to_idxs[int(y)].append(idx)
    for y in range(10):
        random.shuffle(label_to_idxs[y])
    val = []
    for y in range(10):
        val.extend(label_to_idxs[y][:per_class])
    random.shuffle(val)
    return val


# ============================
# Dirichlet non-IID split
# ============================
def make_clients_dirichlet_indices(train_ds, n_clients: int = 20, alpha: float = 0.3, seed: int = 123) -> List[List[int]]:
    rng = np.random.RandomState(seed)

    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx in range(len(train_ds)):
        _, y = train_ds[idx]
        label_to_idxs[int(y)].append(idx)

    for y in range(10):
        rng.shuffle(label_to_idxs[y])

    clients = [[] for _ in range(n_clients)]

    for y in range(10):
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
# Compute deltas + prox + probe_now + dloss_now
# ============================
def compute_deltas_prox_probe_dloss_now(
    model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    local_lr: float,
    local_steps: int,
    probe_batches: int = 2,
    dloss_batches: int = 2,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    gref = server_reference_grad(model, val_loader, batches=10)
    desc = (-gref).detach()
    desc_norm = desc / (desc.norm() + 1e-12)

    deltas: List[torch.Tensor] = []
    norms: List[float] = []
    probe_now: List[float] = []
    dloss_now: List[float] = []

    for loader in client_loaders:
        probe_now.append(float(probing_loss(model, loader, batches=probe_batches)))
        dw, dl = local_train_delta_and_dloss(model, loader, lr=local_lr, steps=local_steps, dloss_batches=dloss_batches)
        deltas.append(dw)
        norms.append(float(dw.norm().item()))
        dloss_now.append(float(dl))

    c = float(np.median(norms) + 1e-12)

    prox: List[float] = []
    for dw in deltas:
        cosv = cosine(dw, desc_norm)
        sat = float(torch.tanh(dw.norm() / c).item())
        prox.append(cosv * sat)

    return (
        deltas,
        np.array(prox, dtype=np.float32),
        np.array(probe_now, dtype=np.float32),
        np.array(dloss_now, dtype=np.float32),
    )


# ============================
# Apply FedAvg
# ============================
def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor], selected: List[int]):
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


# ============================
# EMA updates (probe + dloss)
# ============================
def update_ema(arr_ema: np.ndarray, arr_now: np.ndarray, alpha: float, init_if_zero: bool = True) -> np.ndarray:
    if init_if_zero and np.allclose(arr_ema, 0.0):
        arr_ema[:] = arr_now
        return arr_ema
    arr_ema[:] = (1.0 - alpha) * arr_ema + alpha * arr_now
    return arr_ema


# ============================
# (Optional) Metric selection (your original)
# ============================
def select_by_metric_prox_probeEMA_saturating(
    prox: np.ndarray,
    probe_ema: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
    k: int,
    tau: float = 0.6,
    eps: float = 0.10,
    lam_stale: float = 0.03,
    gamma_streak: float = 0.6,
    q_lo: float = 0.50,
    q_hi: float = 0.90,
    kappa: float = 2.0,
    power: float = 1.0,
) -> List[int]:
    n = len(prox)
    u = np.maximum(0.0, prox).astype(np.float64) + 1e-12

    p = probe_ema.astype(np.float64)
    p0 = float(np.quantile(p, q_lo))
    p1 = float(np.quantile(p, q_hi))
    denom = float((p1 - p0) + 1e-8)

    z = (p - p0) / denom
    z = np.clip(z, 0.0, 1.0)

    bonus01 = 1.0 / (1.0 + np.exp(-kappa * (z - 0.5)))
    bonus01 = (bonus01 - bonus01.min()) / (bonus01.max() - bonus01.min() + 1e-12)
    bonus = 0.25 + 0.75 * bonus01
    bonus = (bonus ** power)

    u = u * bonus
    u = u * (1.0 + lam_stale * staleness)
    u = u * (gamma_streak ** streak)

    logits = u / max(1e-12, tau)
    logits = logits - logits.max()
    psel = np.exp(logits)
    psel = psel / psel.sum()
    psel = (1.0 - eps) * psel + eps * (1.0 / n)

    sel = torch.multinomial(torch.tensor(psel, dtype=torch.float32), num_samples=k, replacement=False).tolist()
    return sel


# ============================
# Context builder (d=6)
# ============================
def build_context_matrix_nn(
    prox: np.ndarray,
    probe_ema: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
    dloss_ema: np.ndarray,
) -> np.ndarray:
    n = len(prox)
    relu_prox = np.maximum(0.0, prox).astype(np.float32)

    p = probe_ema.astype(np.float32)
    pm, ps = float(p.mean()), float(p.std() + 1e-6)
    pz = np.clip((p - pm) / ps, -3.0, 3.0).astype(np.float32)

    s = staleness.astype(np.float32)
    smax = float(s.max() + 1e-6)
    sn = (s / smax).astype(np.float32)

    cap = 5.0
    t = streak.astype(np.float32)
    tn = np.clip(t / cap, 0.0, 1.0).astype(np.float32)

    d = dloss_ema.astype(np.float32)
    dm, ds = float(d.mean()), float(d.std() + 1e-6)
    dz = np.clip((d - dm) / ds, -3.0, 3.0).astype(np.float32)

    X = np.stack([np.ones(n, dtype=np.float32), relu_prox, pz, sn, tn, dz], axis=1)  # (n,6)
    return X


# ============================
# DQN selector (2 actions)
# ============================
class QNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # Q(x,0), Q(x,1)
        )

    def forward(self, x):
        return self.net(x)


class DQNReplay:
    def __init__(self, capacity: int, d_in: int):
        self.capacity = int(capacity)
        self.d_in = int(d_in)
        self.x  = np.zeros((capacity, d_in), dtype=np.float32)
        self.a  = np.zeros((capacity,), dtype=np.int64)
        self.r  = np.zeros((capacity,), dtype=np.float32)
        self.x2 = np.zeros((capacity, d_in), dtype=np.float32)
        self.d  = np.zeros((capacity,), dtype=np.float32)  # done (0/1)
        self.n = 0
        self.ptr = 0

    def add(self, x, a, r, x2, done: bool):
        self.x[self.ptr] = x.astype(np.float32)
        self.a[self.ptr] = int(a)
        self.r[self.ptr] = float(r)
        self.x2[self.ptr] = x2.astype(np.float32)
        self.d[self.ptr] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int):
        bs = min(int(batch_size), self.n)
        idx = np.random.choice(self.n, size=bs, replace=False)
        return self.x[idx], self.a[idx], self.r[idx], self.x2[idx], self.d[idx]


class DQNSelector:
    def __init__(
        self,
        d_in: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.90,
        buf_size: int = 20000,
        batch_size: int = 256,
        train_steps: int = 10,
        grad_clip: float = 1.0,
        target_sync_every: int = 10,
        double_dqn: bool = True,
    ):
        self.d_in = int(d_in)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)

        self.q = QNet(d_in).to(DEVICE)
        self.q_tgt = QNet(d_in).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.buf = DQNReplay(buf_size, d_in)

        self._train_calls = 0

    @torch.no_grad()
    def q_values(self, X: np.ndarray) -> np.ndarray:
        self.q.eval()
        xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        q = self.q(xt).detach().cpu().numpy().astype(np.float64)  # (n,2)
        return q

    def select_topk(self, X: np.ndarray, k: int, eps: float = 0.10) -> List[int]:
        n = X.shape[0]
        if np.random.rand() < eps:
            return np.random.choice(np.arange(n), size=k, replace=False).tolist()
        q = self.q_values(X)
        adv = q[:, 1] - q[:, 0]
        sel = np.argsort(adv)[::-1][:k].tolist()
        return sel

    def add_transition(self, x, a, r, x2, done: bool):
        self.buf.add(x, a, r, x2, done)

    def train(self):
        if self.buf.n < max(256, self.batch_size // 2):
            return

        self.q.train()
        for _ in range(self.train_steps):
            xb, ab, rb, x2b, db = self.buf.sample(self.batch_size)

            x  = torch.tensor(xb,  dtype=torch.float32, device=DEVICE)
            a  = torch.tensor(ab,  dtype=torch.int64,   device=DEVICE).unsqueeze(1)
            r  = torch.tensor(rb,  dtype=torch.float32, device=DEVICE)
            x2 = torch.tensor(x2b, dtype=torch.float32, device=DEVICE)
            d  = torch.tensor(db,  dtype=torch.float32, device=DEVICE)

            q_all = self.q(x)                     # (B,2)
            q_sa  = q_all.gather(1, a).squeeze(1)  # (B,)

            with torch.no_grad():
                if self.double_dqn:
                    a2 = self.q(x2).argmax(dim=1, keepdim=True)         # (B,1)
                    q2 = self.q_tgt(x2).gather(1, a2).squeeze(1)         # (B,)
                else:
                    q2 = self.q_tgt(x2).max(dim=1).values               # (B,)

                y = r + (1.0 - d) * self.gamma * q2

            loss = F.smooth_l1_loss(q_sa, y)

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())


# ============================
# Experiment runner
# ============================
def run_experiment(
    rounds: int = 80,
    n_clients: int = 20,
    k_select: int = 5,
    dir_alpha: float = 0.3,
    flip_fraction: float = 0.50,
    flip_rate: float = 0.80,
    max_per_client: int = 3000,
    local_lr: float = 0.05,
    local_steps: int = 40,
    probe_batches: int = 2,
    dloss_batches: int = 2,
    ema_alpha: float = 0.10,
    print_every: int = 10,
    # DQN params
    dqn_eps: float = 0.10,
    dqn_lr: float = 1e-3,
    dqn_gamma: float = 0.90,
    dqn_train_steps: int = 10,
    dqn_batch_size: int = 256,
    dqn_buf_size: int = 20000,
    dqn_target_sync_every: int = 10,
    neg_ratio: int = 1,  # how many "a=0, r=0" clients to log per selected client (per round)
    use_metric_track: bool = True,
):
    tfm = transforms.Compose([transforms.ToTensor()])

    log_step("Baixando/carregando MNIST...")
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    # server loaders
    val_idxs = make_server_val_balanced(test_ds, per_class=200)  # 2000 samples
    val_loader = DataLoader(
        Subset(test_ds, val_idxs),
        batch_size=256,
        shuffle=True,
        generator=g_dl,
        worker_init_fn=seed_worker,
        num_workers=0,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # clients split
    log_step(f"Gerando split Dirichlet (alpha={dir_alpha}) para {n_clients} clientes...")
    client_idxs = make_clients_dirichlet_indices(train_ds, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777)

    # fixed poisoned set
    n_flip = int(round(flip_fraction * n_clients))
    rng = np.random.RandomState(SEED + 999)
    flip_clients = set(rng.choice(np.arange(n_clients), size=n_flip, replace=False).tolist())

    client_loaders: List[DataLoader] = []
    client_sizes: List[int] = []

    log_step("Criando loaders dos clientes + (opcional) label flipping...")
    for cid, idxs in enumerate(client_idxs):
        if max_per_client is not None:
            idxs = idxs[:max_per_client]
        client_sizes.append(len(idxs))

        if cid in flip_clients:
            ds_c = LabelFlipSubset(train_ds, idxs, flip_rate=flip_rate, n_classes=10, seed=SEED + 1000 + cid)
        else:
            ds_c = Subset(train_ds, idxs)

        client_loaders.append(
            DataLoader(
                ds_c,
                batch_size=64,
                shuffle=True,
                generator=g_dl,
                worker_init_fn=seed_worker,
                num_workers=0,
            )
        )

    # base init
    base = MLP().to(DEVICE)

    # Track A: RANDOM
    model_rand = copy.deepcopy(base).to(DEVICE)
    staleness_r = np.zeros(n_clients, dtype=np.float32)
    streak_r = np.zeros(n_clients, dtype=np.int32)
    probe_ema_r = np.zeros(n_clients, dtype=np.float32)
    dloss_ema_r = np.zeros(n_clients, dtype=np.float32)

    # Track B: DQN selector
    model_dqn = copy.deepcopy(base).to(DEVICE)
    staleness_b = np.zeros(n_clients, dtype=np.float32)
    streak_b = np.zeros(n_clients, dtype=np.int32)
    probe_ema_b = np.zeros(n_clients, dtype=np.float32)
    dloss_ema_b = np.zeros(n_clients, dtype=np.float32)
    dqn = DQNSelector(
        d_in=6,
        lr=dqn_lr,
        weight_decay=1e-4,
        gamma=dqn_gamma,
        buf_size=dqn_buf_size,
        batch_size=dqn_batch_size,
        train_steps=dqn_train_steps,
        grad_clip=1.0,
        target_sync_every=dqn_target_sync_every,
        double_dqn=True,
    )

    # Track C: METRIC (optional)
    if use_metric_track:
        model_met = copy.deepcopy(base).to(DEVICE)
        staleness_m = np.zeros(n_clients, dtype=np.float32)
        streak_m = np.zeros(n_clients, dtype=np.int32)
        probe_ema_m = np.zeros(n_clients, dtype=np.float32)
        dloss_ema_m = np.zeros(n_clients, dtype=np.float32)

    print(f"\nDEVICE={DEVICE}")
    print(f"N_CLIENTS={n_clients} | K={k_select} | rounds={rounds} | dir_alpha={dir_alpha}")
    print(f"Flip: fraction_clients={flip_fraction} -> n_flip={n_flip} | flip_rate_samples={flip_rate}")
    print(f"Flip clients (fixed): {sorted(list(flip_clients))}")
    print(f"Avg client size (capped) ~ {np.mean(client_sizes):.1f} samples")
    print(f"DQN: eps={dqn_eps}, gamma={dqn_gamma}, train_steps={dqn_train_steps}, target_sync_every={dqn_target_sync_every}")
    print("Comparando: RANDOM vs DQN" + (" vs METRIC" if use_metric_track else "") + "\n")

    # Pending transitions for DQN: list of tuples (cid, action, x, reward)
    pending_prev: List[Tuple[int, int, np.ndarray, float]] = []
    X_prev: np.ndarray = None  # for sanity, not required

    for t in range(1, rounds + 1):
        t0 = time.time()
        log_step(f"\n[round {t}/{rounds}] começando...")

        # ---- evaluation BEFORE updates
        l_r_before = eval_loss(model_rand, val_loader, max_batches=20)
        a_r = eval_acc(model_rand, test_loader, max_batches=80)

        l_b_before = eval_loss(model_dqn, val_loader, max_batches=20)
        a_b = eval_acc(model_dqn, test_loader, max_batches=80)

        if use_metric_track:
            l_m_before = eval_loss(model_met, val_loader, max_batches=20)
            a_m = eval_acc(model_met, test_loader, max_batches=80)
            log_step(
                f"  eval | rand(loss={l_r_before:.4f}, acc={a_r*100:.2f}%) | "
                f"dqn(loss={l_b_before:.4f}, acc={a_b*100:.2f}%) | "
                f"met(loss={l_m_before:.4f}, acc={a_m*100:.2f}%)"
            )
        else:
            log_step(
                f"  eval | rand(loss={l_r_before:.4f}, acc={a_r*100:.2f}%) | "
                f"dqn(loss={l_b_before:.4f}, acc={a_b*100:.2f}%)"
            )

        # =========================================================
        # Track A: RANDOM
        # =========================================================
        deltas_r, prox_r, probe_now_r, dloss_now_r = compute_deltas_prox_probe_dloss_now(
            model_rand,
            client_loaders,
            val_loader,
            local_lr,
            local_steps,
            probe_batches=probe_batches,
            dloss_batches=dloss_batches,
        )
        update_ema(probe_ema_r, probe_now_r, alpha=ema_alpha, init_if_zero=True)
        update_ema(dloss_ema_r, dloss_now_r, alpha=ema_alpha, init_if_zero=True)

        selected_r = random.sample(range(n_clients), k_select)
        apply_fedavg(model_rand, deltas_r, selected_r)
        update_staleness_streak(staleness_r, streak_r, selected_r)

        l_r_after = eval_loss(model_rand, val_loader, max_batches=20)
        r_reward = float(l_r_before - l_r_after)
        log_step(f"  RANDOM: sel={selected_r} | reward={r_reward:+.5f} (val {l_r_before:.4f}->{l_r_after:.4f})")

        # =========================================================
        # Track B: DQN selector
        # =========================================================
        deltas_b, prox_b, probe_now_b, dloss_now_b = compute_deltas_prox_probe_dloss_now(
            model_dqn,
            client_loaders,
            val_loader,
            local_lr,
            local_steps,
            probe_batches=probe_batches,
            dloss_batches=dloss_batches,
        )
        update_ema(probe_ema_b, probe_now_b, alpha=ema_alpha, init_if_zero=True)
        update_ema(dloss_ema_b, dloss_now_b, alpha=ema_alpha, init_if_zero=True)

        Xb = build_context_matrix_nn(prox_b, probe_ema_b, staleness_b, streak_b, dloss_ema_b)

        # ---- First: if we have pending transitions from previous round, close them using x_{t} as next state
        if pending_prev:
            for (cid, a, x, r) in pending_prev:
                x2 = Xb[cid]
                dqn.add_transition(x=x, a=a, r=r, x2=x2, done=False)

        # ---- Select clients with DQN
        selected_b = dqn.select_topk(Xb, k=k_select, eps=dqn_eps)
        apply_fedavg(model_dqn, deltas_b, selected_b)
        update_staleness_streak(staleness_b, streak_b, selected_b)

        l_b_after = eval_loss(model_dqn, val_loader, max_batches=20)
        b_reward = float(l_b_before - l_b_after)

        # credit assignment (simple)
        per_client_r = b_reward / max(1, len(selected_b))

        # also log some unselected with a=0, r=0 to help learn Q(x,0)
        not_sel = [i for i in range(n_clients) if i not in set(selected_b)]
        n_neg = min(len(not_sel), neg_ratio * k_select)
        neg_cids = random.sample(not_sel, n_neg) if n_neg > 0 else []

        # create pending transitions for THIS round (to be closed next round)
        pending_prev = []
        for cid in selected_b:
            pending_prev.append((cid, 1, Xb[cid].copy(), float(per_client_r)))
        for cid in neg_cids:
            pending_prev.append((cid, 0, Xb[cid].copy(), 0.0))

        # train DQN now (using transitions we just added from the previous pending)
        dqn.train()

        log_step(f"  DQN: sel={selected_b} | reward={b_reward:+.5f} (val {l_b_before:.4f}->{l_b_after:.4f}) | neg={len(neg_cids)}")

        # =========================================================
        # Track C: METRIC (optional)
        # =========================================================
        if use_metric_track:
            deltas_m, prox_m, probe_now_m, dloss_now_m = compute_deltas_prox_probe_dloss_now(
                model_met,
                client_loaders,
                val_loader,
                local_lr,
                local_steps,
                probe_batches=probe_batches,
                dloss_batches=dloss_batches,
            )
            update_ema(probe_ema_m, probe_now_m, alpha=ema_alpha, init_if_zero=True)
            update_ema(dloss_ema_m, dloss_now_m, alpha=ema_alpha, init_if_zero=True)

            selected_m = select_by_metric_prox_probeEMA_saturating(
                prox=prox_m,
                probe_ema=probe_ema_m,
                staleness=staleness_m,
                streak=streak_m,
                k=k_select,
            )
            apply_fedavg(model_met, deltas_m, selected_m)
            update_staleness_streak(staleness_m, streak_m, selected_m)

            l_m_after = eval_loss(model_met, val_loader, max_batches=20)
            m_reward = float(l_m_before - l_m_after)
            log_step(f"  METRIC: sel={selected_m} | reward={m_reward:+.5f} (val {l_m_before:.4f}->{l_m_after:.4f})")

        log_step(f"[round {t}/{rounds}] terminado em {time.time()-t0:.2f}s")

        if t % print_every == 0:
            if use_metric_track:
                print(
                    f"[summary @ {t:3d}] "
                    f"RAND loss={l_r_before:.4f} r={r_reward:+.4f} | "
                    f"DQN  loss={l_b_before:.4f} r={b_reward:+.4f} | "
                    f"MET  loss={l_m_before:.4f} r={m_reward:+.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[summary @ {t:3d}] "
                    f"RAND loss={l_r_before:.4f} r={r_reward:+.4f} | "
                    f"DQN  loss={l_b_before:.4f} r={b_reward:+.4f}",
                    flush=True,
                )

    # flush last pending as terminal (done=True, target becomes y=r)
    if pending_prev:
        for (cid, a, x, r) in pending_prev:
            dqn.add_transition(x=x, a=a, r=r, x2=x, done=True)
        dqn.train()

    print("\nDone.")


if __name__ == "__main__":
    run_experiment(
        rounds=80,
        n_clients=20,
        k_select=5,
        dir_alpha=0.3,
        flip_fraction=0.50,
        flip_rate=1,
        max_per_client=3000,
        local_lr=0.05,
        local_steps=40,
        probe_batches=2,
        dloss_batches=2,
        ema_alpha=0.10,
        print_every=10,
        dqn_eps=0.10,
        dqn_lr=1e-3,
        dqn_gamma=0.90,
        dqn_train_steps=10,
        dqn_batch_size=256,
        dqn_buf_size=20000,
        dqn_target_sync_every=10,
        neg_ratio=1,          # try 1 or 2
        use_metric_track=False,
    )