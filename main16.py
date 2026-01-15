"""
FL client selection experiment (MNIST) — Random vs Metric
UPDATED: adds per-client Probing EMA (stabilizes probing measured on few batches)

Tracks (two parallel worlds):
  (A) RANDOM: select K clients uniformly at random each round
  (B) METRIC: select K clients using:
        prox(i)  = alignment(Δw_i, -g_ref) * tanh(||Δw_i||/c)
        probeEMA(i) = EMA of probing loss (global model on client data, few batches)
        score(i) = ReLU(prox(i)) * bonus(probeEMA(i))   (bonus increasing, saturating)
      plus staleness + streak + epsilon exploration.

Setup:
  - 20 clients, Dirichlet non-IID split
  - A fraction of clients is poisoned (fixed set) with label flipping on a fraction of their samples
  - K=5 selected per round
  - Server uses balanced validation set (IID-ish) to compute g_ref

Reproducibility:
  - Seeds fixed, deterministic cuDNN toggles set
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
# Model
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
    """
    Probing loss = CE loss of CURRENT global model on a small sample of client data.
    Cheap + noisy. We'll smooth with EMA across rounds.
    """
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
# Server reference gradient
# ============================
def server_reference_grad(model: nn.Module, val_loader: DataLoader, batches: int = 10) -> torch.Tensor:
    """
    g_ref = ∇ L_val(w) (points to INCREASE loss). Descent direction is -g_ref.
    """
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
def local_train_delta(global_model: nn.Module, loader: DataLoader, lr: float = 0.05, steps: int = 30) -> torch.Tensor:
    """
    Train a copy locally and return Δw = w_local - w_global.
    """
    model = copy.deepcopy(global_model).to(DEVICE)
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

    w1 = flatten_params(model)
    return (w1 - w0).detach()


# ============================
# Label flipping wrapper
# ============================
class LabelFlipSubset(torch.utils.data.Dataset):
    """
    Subset that flips labels with probability flip_rate per sample access.
    (stochastic/online flip)
    """
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
# Compute deltas + prox + probe_now
# ============================
def compute_deltas_prox_probe_now(
    model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    local_lr: float,
    local_steps: int,
    probe_batches: int = 2,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    """
    Returns:
      deltas: list of Δw_i
      prox:   prox_i = cos(Δw_i, -gref) * tanh(||Δw_i||/c)
      probe_now: probing loss on few batches for each client
    """
    gref = server_reference_grad(model, val_loader, batches=10)
    desc = (-gref).detach()
    desc_norm = desc / (desc.norm() + 1e-12)

    deltas: List[torch.Tensor] = []
    norms: List[float] = []
    probe_now: List[float] = []

    for loader in client_loaders:
        probe_now.append(float(probing_loss(model, loader, batches=probe_batches)))
        dw = local_train_delta(model, loader, lr=local_lr, steps=local_steps)
        deltas.append(dw)
        norms.append(float(dw.norm().item()))

    c = float(np.median(norms) + 1e-12)

    prox: List[float] = []
    for dw in deltas:
        cosv = cosine(dw, desc_norm)
        sat = float(torch.tanh(dw.norm() / c).item())
        prox.append(cosv * sat)

    return deltas, np.array(prox, dtype=np.float32), np.array(probe_now, dtype=np.float32)


# ============================
# Apply FedAvg
# ============================
def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor], selected: List[int]):
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


# ============================
# Metric selection with probing EMA (bonus increasing + saturating)
# ============================
def select_by_metric_prox_probeEMA_saturating(
    prox: np.ndarray,
    probe_ema: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
    k: int,
    # sampling / exploration
    tau: float = 0.6,
    eps: float = 0.10,
    # diversity controls
    lam_stale: float = 0.03,
    gamma_streak: float = 0.6,
    # probing bonus controls
    q_lo: float = 0.50,    # baseline quantile (median)
    q_hi: float = 0.90,    # saturation quantile
    kappa: float = 2.0,    # steepness of bonus curve
    power: float = 1.0,    # how much probing influences
) -> List[int]:
    """
    score_i = ReLU(prox_i) * bonus(probeEMA_i)
    bonus(probeEMA) increases up to q_hi then saturates (NO penalty for very high probe).
    """
    n = len(prox)

    # gate by direction
    u = np.maximum(0.0, prox).astype(np.float64) + 1e-12

    # quantile anchors on probe_ema
    p = probe_ema.astype(np.float64)
    p0 = float(np.quantile(p, q_lo))
    p1 = float(np.quantile(p, q_hi))
    denom = float((p1 - p0) + 1e-8)

    z = (p - p0) / denom
    z = np.clip(z, 0.0, 1.0)

    # smooth increasing bonus, normalized to [0,1], then mapped to [0.25, 1.0]
    bonus01 = 1.0 / (1.0 + np.exp(-kappa * (z - 0.5)))
    bonus01 = (bonus01 - bonus01.min()) / (bonus01.max() - bonus01.min() + 1e-12)
    bonus = 0.25 + 0.75 * bonus01
    bonus = bonus ** power

    u = u * bonus

    # staleness / streak shaping
    u = u * (1.0 + lam_stale * staleness)
    u = u * (gamma_streak ** streak)

    # softmax sampling
    logits = u / max(1e-12, tau)
    logits = logits - logits.max()
    psel = np.exp(logits)
    psel = psel / psel.sum()

    # epsilon mix
    psel = (1.0 - eps) * psel + eps * (1.0 / n)

    sel = torch.multinomial(torch.tensor(psel, dtype=torch.float32), num_samples=k, replacement=False).tolist()
    return sel


# ============================
# EMA update
# ============================
def update_probe_ema(probe_ema: np.ndarray, probe_now: np.ndarray, alpha: float, init_if_zero: bool = True) -> np.ndarray:
    """
    probe_ema <- (1-alpha)*probe_ema + alpha*probe_now
    If init_if_zero: when probe_ema is zero (first round), set to probe_now to avoid warmup bias.
    """
    if init_if_zero and np.allclose(probe_ema, 0.0):
        probe_ema[:] = probe_now
        return probe_ema
    probe_ema[:] = (1.0 - alpha) * probe_ema + alpha * probe_now
    return probe_ema


# ============================
# Experiment runner
# ============================
def run_experiment(
    rounds: int = 80,
    n_clients: int = 20,
    k_select: int = 5,
    dir_alpha: float = 0.3,
    flip_fraction: float = 0.30,  # fraction of clients poisoned
    flip_rate: float = 0.30,      # fraction of samples flipped within poisoned clients
    max_per_client: int = 3000,
    local_lr: float = 0.05,
    local_steps: int = 40,
    probe_batches: int = 2,
    probe_ema_alpha: float = 0.10,  # EMA smoothing
    print_every: int = 10,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    # clients
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
            ds_c = LabelFlipSubset(
                base_ds=train_ds,
                indices=idxs,
                flip_rate=flip_rate,
                n_classes=10,
                seed=SEED + 1000 + cid,
            )
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

    # two models (same init)
    base = MLP().to(DEVICE)
    model_rand = copy.deepcopy(base).to(DEVICE)
    model_met = copy.deepcopy(base).to(DEVICE)

    # staleness/streak for each track
    staleness_r = np.zeros(n_clients, dtype=np.float32)
    streak_r = np.zeros(n_clients, dtype=np.int32)
    staleness_m = np.zeros(n_clients, dtype=np.float32)
    streak_m = np.zeros(n_clients, dtype=np.int32)

    # probing EMA per client for each track (because models diverge)
    probe_ema_r = np.zeros(n_clients, dtype=np.float32)
    probe_ema_m = np.zeros(n_clients, dtype=np.float32)

    hist = {"rand_loss": [], "rand_acc": [], "met_loss": [], "met_acc": []}

    print(f"\nDEVICE={DEVICE}")
    print(f"N_CLIENTS={n_clients} | K={k_select} | rounds={rounds} | dir_alpha={dir_alpha}")
    print(f"Flip: fraction_clients={flip_fraction} -> n_flip={n_flip} | flip_rate_samples={flip_rate}")
    print(f"Flip clients (fixed): {sorted(list(flip_clients))}")
    print(f"Avg client size (capped) ~ {np.mean(client_sizes):.1f} samples")
    print(f"Probing: {probe_batches} batches/round, EMA alpha={probe_ema_alpha}")
    print("Comparando: RANDOM vs MÉTRICA(ReLU(prox)*bonus(probeEMA) + staleness + streak + eps)\n")

    for t in range(1, rounds + 1):
        t0 = time.time()
        log_step(f"\n[round {t}/{rounds}] começando...")

        # ---- evaluation
        log_step("  - avaliando modelos no servidor (val loss / test acc)...")
        ta = time.time()
        l_r = eval_loss(model_rand, val_loader, max_batches=20)
        a_r = eval_acc(model_rand, test_loader, max_batches=80)
        l_m = eval_loss(model_met, val_loader, max_batches=20)
        a_m = eval_acc(model_met, test_loader, max_batches=80)
        log_step(
            f"    ok (eval) em {time.time()-ta:.2f}s | "
            f"rand(loss={l_r:.4f}, acc={a_r*100:.2f}%) | "
            f"met(loss={l_m:.4f}, acc={a_m*100:.2f}%)"
        )

        hist["rand_loss"].append(l_r)
        hist["rand_acc"].append(a_r)
        hist["met_loss"].append(l_m)
        hist["met_acc"].append(a_m)

        # ---- RANDOM track
        log_step("  - RANDOM: calculando deltas + prox + probe_now (todos os clientes)...")
        tb = time.time()
        deltas_r, prox_r, probe_now_r = compute_deltas_prox_probe_now(
            model_rand, client_loaders, val_loader, local_lr, local_steps, probe_batches=probe_batches
        )
        update_probe_ema(probe_ema_r, probe_now_r, alpha=probe_ema_alpha, init_if_zero=True)
        log_step(f"    ok (random stats) em {time.time()-tb:.2f}s")

        selected_r = random.sample(range(n_clients), k_select)
        apply_fedavg(model_rand, deltas_r, selected_r)
        update_staleness_streak(staleness_r, streak_r, selected_r)
        log_step(f"    RANDOM: selecionados {selected_r} | FedAvg aplicado")

        # ---- METRIC track
        log_step("  - MET: calculando deltas + prox + probe_now (todos os clientes)...")
        tc = time.time()
        deltas_m, prox_m, probe_now_m = compute_deltas_prox_probe_now(
            model_met, client_loaders, val_loader, local_lr, local_steps, probe_batches=probe_batches
        )
        update_probe_ema(probe_ema_m, probe_now_m, alpha=probe_ema_alpha, init_if_zero=True)
        log_step(f"    ok (metric stats) em {time.time()-tc:.2f}s")

        log_step("    MET: selecionando K clientes usando prox + probeEMA (bonus saturado)...")
        selected_m = select_by_metric_prox_probeEMA_saturating(
            prox=prox_m,
            probe_ema=probe_ema_m,
            staleness=staleness_m,
            streak=streak_m,
            k=k_select,
            tau=0.6,
            eps=0.10,
            lam_stale=0.03,
            gamma_streak=0.6,
            q_lo=0.50,
            q_hi=0.90,
            kappa=2.0,
            power=1.0,
        )
        apply_fedavg(model_met, deltas_m, selected_m)
        update_staleness_streak(staleness_m, streak_m, selected_m)

        top_prox = np.argsort(prox_m)[::-1][:5].tolist()
        top_probe = np.argsort(probe_ema_m)[::-1][:5].tolist()
        log_step(
            f"    MET: selecionados {selected_m} | top5 prox={top_prox} | top5 probeEMA={top_probe}"
        )

        log_step(f"[round {t}/{rounds}] terminado em {time.time()-t0:.2f}s")

        if t % print_every == 0:
            print(
                f"[summary @ {t:3d}] RAND loss={l_r:.4f} acc={a_r*100:.2f}% | "
                f"MET loss={l_m:.4f} acc={a_m*100:.2f}%",
                flush=True,
            )

    print("\nFinais:")
    print(f"RAND final: loss_val={hist['rand_loss'][-1]:.4f} acc_test={hist['rand_acc'][-1]*100:.2f}%")
    print(f"MET  final: loss_val={hist['met_loss'][-1]:.4f} acc_test={hist['met_acc'][-1]*100:.2f}%")

    return hist


if __name__ == "__main__":
    run_experiment(
        rounds=80,
        n_clients=20,
        k_select=5,
        dir_alpha=0.3,         # smaller => more non-IID
        flip_fraction=0.50,    # 30% poisoned clients
        flip_rate=0.80,        # 30% samples flipped within those clients
        max_per_client=3000,
        local_lr=0.05,
        local_steps=40,
        probe_batches=2,       # few batches -> cheap; EMA reduces noise
        probe_ema_alpha=0.10,  # try 0.2 if you want faster reaction
        print_every=10,
    )