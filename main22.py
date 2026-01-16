# fl_dqn_per_cifar10_stable_reward.py
# Federated client selection on CIFAR-10:
#   Track A: RANDOM
#   Track B: DQN (Double DQN + Target Net) + PER
#
# Key changes for stabler reward:
# - val_loader: shuffle=False so "before" and "after" compare the same batches
# - eval_loss/eval_acc default to full-loader evaluation (less noisy reward)
#
# Run:
#   python fl_dqn_per_cifar10_stable_reward.py

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

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
# Model (CIFAR-10 CNN)
# ============================
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32->16

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16->8
        self.pool3 = nn.MaxPool2d(2, 2)  # 8->4

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))     # -> (64,16,16)
        x = self.pool2(F.relu(self.conv3(x)))    # -> (128,8,8)
        x = self.pool3(x)                        # -> (128,4,4)
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
def eval_loss(model: nn.Module, loader: DataLoader, max_batches: Optional[int] = None) -> float:
    model.eval()
    total = 0.0
    n = 0
    for b, (x, y) in enumerate(loader):
        if (max_batches is not None) and (b >= max_batches):
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, max_batches: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for b, (x, y) in enumerate(loader):
        if (max_batches is not None) and (b >= max_batches):
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


@torch.no_grad()
def loss_on_few_batches(model: nn.Module, loader: DataLoader, batches: int = 1) -> float:
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
    lr: float = 0.01,
    steps: int = 10,
    dloss_batches: int = 1,
) -> Tuple[torch.Tensor, float]:
    model = copy.deepcopy(global_model).to(DEVICE)

    loss_before = loss_on_few_batches(model, loader, batches=dloss_batches)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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
def make_server_val_balanced(ds, per_class: int = 200, n_classes: int = 10) -> List[int]:
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        label_to_idxs[int(y)].append(idx)
    for y in range(n_classes):
        random.shuffle(label_to_idxs[y])
    val = []
    for y in range(n_classes):
        val.extend(label_to_idxs[y][:per_class])
    random.shuffle(val)
    return val


# ============================
# Dirichlet non-IID split
# ============================
def make_clients_dirichlet_indices(
    train_ds,
    n_clients: int = 20,
    alpha: float = 0.3,
    seed: int = 123,
    n_classes: int = 10
) -> List[List[int]]:
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
    probe_batches: int = 1,
    dloss_batches: int = 1,
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
        dw, dl = local_train_delta_and_dloss(
            model, loader, lr=local_lr, steps=local_steps, dloss_batches=dloss_batches
        )
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
# DQN selector (2 actions) + PER
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


class PrioritizedReplay:
    """
    PER proportional:
      p_i = (|td_i| + eps)
    Sampling probs:
      P(i) ∝ (p_i)^alpha
    IS weights:
      w_i = (N * P(i))^{-beta} / max_w
    """
    def __init__(self, capacity: int, d_in: int, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = int(capacity)
        self.d_in = int(d_in)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.x  = np.zeros((capacity, d_in), dtype=np.float32)
        self.a  = np.zeros((capacity,), dtype=np.int64)
        self.r  = np.zeros((capacity,), dtype=np.float32)
        self.x2 = np.zeros((capacity, d_in), dtype=np.float32)
        self.d  = np.zeros((capacity,), dtype=np.float32)

        self.p  = np.zeros((capacity,), dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.max_p = 1.0

    def add(self, x, a, r, x2, done: bool):
        self.x[self.ptr] = x.astype(np.float32)
        self.a[self.ptr] = int(a)
        self.r[self.ptr] = float(r)
        self.x2[self.ptr] = x2.astype(np.float32)
        self.d[self.ptr] = 1.0 if done else 0.0

        self.p[self.ptr] = self.max_p
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        bs = min(int(batch_size), self.n)
        assert bs > 0

        pri = self.p[:self.n].astype(np.float64)
        probs = (pri + self.eps) ** self.alpha
        probs = probs / probs.sum()

        idx = np.random.choice(self.n, size=bs, replace=False, p=probs)

        N = self.n
        w = (N * probs[idx]) ** (-beta)
        w = w / (w.max() + 1e-12)

        return (
            self.x[idx],
            self.a[idx],
            self.r[idx],
            self.x2[idx],
            self.d[idx],
            idx,
            w.astype(np.float32),
        )

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.p[idx] = td_abs + self.eps
        self.max_p = float(max(self.max_p, float(td_abs.max(initial=0.0))))


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
        # PER
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 2000,
        per_eps: float = 1e-3,
    ):
        self.d_in = int(d_in)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)

        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = int(per_beta_steps)

        self.q = QNet(d_in).to(DEVICE)
        self.q_tgt = QNet(d_in).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.buf = PrioritizedReplay(buf_size, d_in, alpha=per_alpha, eps=per_eps)

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

    def _beta(self) -> float:
        t = min(self._train_calls, self.per_beta_steps)
        frac = t / max(1, self.per_beta_steps)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def train(self):
        if self.buf.n < max(256, self.batch_size // 2):
            return

        beta = self._beta()
        self.q.train()

        for _ in range(self.train_steps):
            xb, ab, rb, x2b, db, idx, w_is = self.buf.sample(self.batch_size, beta=beta)

            x  = torch.tensor(xb,  dtype=torch.float32, device=DEVICE)
            a  = torch.tensor(ab,  dtype=torch.int64,   device=DEVICE).unsqueeze(1)
            r  = torch.tensor(rb,  dtype=torch.float32, device=DEVICE)
            x2 = torch.tensor(x2b, dtype=torch.float32, device=DEVICE)
            d  = torch.tensor(db,  dtype=torch.float32, device=DEVICE)
            w  = torch.tensor(w_is, dtype=torch.float32, device=DEVICE)

            q_all = self.q(x)
            q_sa  = q_all.gather(1, a).squeeze(1)

            with torch.no_grad():
                if self.double_dqn:
                    a2 = self.q(x2).argmax(dim=1, keepdim=True)
                    q2 = self.q_tgt(x2).gather(1, a2).squeeze(1)
                else:
                    q2 = self.q_tgt(x2).max(dim=1).values
                y = r + (1.0 - d) * self.gamma * q2

            td = (q_sa - y)
            td_abs = td.detach().abs().cpu().numpy()

            per_sample = F.smooth_l1_loss(q_sa, y, reduction="none")
            loss = (w * per_sample).mean()

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

            self.buf.update_priorities(idx, td_abs)

        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())


# ============================
# Experiment runner
# ============================
def run_experiment(
    rounds: int = 50,
    n_clients: int = 20,
    k_select: int = 5,
    dir_alpha: float = 0.3,
    flip_fraction: float = 0.50,
    flip_rate: float = 0.80,
    max_per_client: int = 2500,
    local_lr: float = 0.01,
    local_steps: int = 10,
    probe_batches: int = 1,
    dloss_batches: int = 1,
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
    neg_ratio: int = 1,
    # PER params
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_steps: int = 2000,
    # Eval control
    val_max_batches: Optional[int] = None,   # None = full val loader (stable)
    test_max_batches: Optional[int] = None,  # None = full test loader
):
    # CIFAR-10 normalization (common values)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    log_step("Baixando/carregando CIFAR-10...")
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    # server loaders
    val_idxs = make_server_val_balanced(test_ds, per_class=200, n_classes=10)  # 2000 samples
    # IMPORTANT: shuffle=False for stable reward
    val_loader = DataLoader(
        Subset(test_ds, val_idxs),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    # test loader can be shuffle=False as usual
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # clients split
    log_step(f"Gerando split Dirichlet (alpha={dir_alpha}) para {n_clients} clientes...")
    client_idxs = make_clients_dirichlet_indices(
        train_ds, n_clients=n_clients, alpha=dir_alpha, seed=SEED + 777, n_classes=10
    )

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
    base = SmallCNN().to(DEVICE)

    # Track A: RANDOM
    model_rand = copy.deepcopy(base).to(DEVICE)
    staleness_r = np.zeros(n_clients, dtype=np.float32)
    streak_r = np.zeros(n_clients, dtype=np.int32)
    probe_ema_r = np.zeros(n_clients, dtype=np.float32)
    dloss_ema_r = np.zeros(n_clients, dtype=np.float32)

    # Track B: DQN selector (Double + PER)
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
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        per_beta_steps=per_beta_steps,
        per_eps=1e-3,
    )

    print(f"\nDEVICE={DEVICE}")
    print(f"N_CLIENTS={n_clients} | K={k_select} | rounds={rounds} | dir_alpha={dir_alpha}")
    print(f"Flip: fraction_clients={flip_fraction} -> n_flip={n_flip} | flip_rate_samples={flip_rate}")
    print(f"Flip clients (fixed): {sorted(list(flip_clients))}")
    print(f"Avg client size (capped) ~ {np.mean(client_sizes):.1f} samples")
    print(f"DQN: eps={dqn_eps}, gamma={dqn_gamma}, train_steps={dqn_train_steps}, target_sync_every={dqn_target_sync_every}")
    print(f"PER: alpha={per_alpha}, beta={per_beta_start}->{per_beta_end} over {per_beta_steps} train() calls")
    print(f"Eval: val_max_batches={val_max_batches} | test_max_batches={test_max_batches}")
    print("Comparando: RANDOM vs DQN(Double+PER)\n")

    pending_prev: List[Tuple[int, int, np.ndarray, float]] = []

    for t in range(1, rounds + 1):
        t0 = time.time()
        log_step(f"\n[round {t}/{rounds}] começando...")

        # ---- evaluation BEFORE updates (stable if val_loader shuffle=False)
        l_r_before = eval_loss(model_rand, val_loader, max_batches=val_max_batches)
        a_r = eval_acc(model_rand, test_loader, max_batches=test_max_batches)

        l_b_before = eval_loss(model_dqn, val_loader, max_batches=val_max_batches)
        a_b = eval_acc(model_dqn, test_loader, max_batches=test_max_batches)

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

        l_r_after = eval_loss(model_rand, val_loader, max_batches=val_max_batches)
        r_reward = float(l_r_before - l_r_after)
        log_step(f"  RANDOM: sel={selected_r} | reward={r_reward:+.5f} (val {l_r_before:.4f}->{l_r_after:.4f})")

        # =========================================================
        # Track B: DQN selector (Double + PER)
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

        # Close pending transitions from previous round using x_t as next state
        if pending_prev:
            for (cid, a, x, r) in pending_prev:
                x2 = Xb[cid]
                dqn.add_transition(x=x, a=a, r=r, x2=x2, done=False)

        # Select clients with DQN
        selected_b = dqn.select_topk(Xb, k=k_select, eps=dqn_eps)
        apply_fedavg(model_dqn, deltas_b, selected_b)
        update_staleness_streak(staleness_b, streak_b, selected_b)

        l_b_after = eval_loss(model_dqn, val_loader, max_batches=val_max_batches)
        b_reward = float(l_b_before - l_b_after)

        # Simple credit assignment
        per_client_r = b_reward / max(1, len(selected_b))

        # Add some negative examples (a=0, r=0) from unselected clients
        not_sel = [i for i in range(n_clients) if i not in set(selected_b)]
        n_neg = min(len(not_sel), neg_ratio * k_select)
        neg_cids = random.sample(not_sel, n_neg) if n_neg > 0 else []

        # Create pending transitions for THIS round
        pending_prev = []
        for cid in selected_b:
            pending_prev.append((cid, 1, Xb[cid].copy(), float(per_client_r)))
        for cid in neg_cids:
            pending_prev.append((cid, 0, Xb[cid].copy(), 0.0))

        # Train DQN (PER updates priorities internally)
        dqn.train()

        log_step(f"  DQN: sel={selected_b} | reward={b_reward:+.5f} (val {l_b_before:.4f}->{l_b_after:.4f}) | neg={len(neg_cids)}")
        log_step(f"[round {t}/{rounds}] terminado em {time.time()-t0:.2f}s")

        if t % print_every == 0:
            print(
                f"[summary @ {t:3d}] "
                f"RAND loss={l_r_before:.4f} r={r_reward:+.4f} | "
                f"DQN  loss={l_b_before:.4f} r={b_reward:+.4f}",
                flush=True,
            )

    # Flush last pending transitions as terminal
    if pending_prev:
        for (cid, a, x, r) in pending_prev:
            dqn.add_transition(x=x, a=a, r=r, x2=x, done=True)
        dqn.train()

    print("\nDone.")


if __name__ == "__main__":
    run_experiment(
        rounds=50,
        n_clients=20,
        k_select=5,
        dir_alpha=0.3,
        flip_fraction=0.50,
        flip_rate=1.0,        # set 0.0 to disable
        max_per_client=2500,
        local_lr=0.01,
        local_steps=10,
        probe_batches=1,
        dloss_batches=1,
        ema_alpha=0.10,
        print_every=10,
        dqn_eps=0.10,
        dqn_lr=1e-3,
        dqn_gamma=0.90,
        dqn_train_steps=10,
        dqn_batch_size=256,
        dqn_buf_size=20000,
        dqn_target_sync_every=10,
        neg_ratio=1,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=2000,
        # evaluation: stable reward
        val_max_batches=None,   # full val (2000 samples) -> stable
        test_max_batches=None,  # full test
    )