# ablation_probe_selection_cifar10_v2.py
# A/B test em FL (CIFAR-10): RANDOM vs seleção por PROBING LOSS (high/low/mid) + opção de pular RANDOM.
#
# Correções/adições nesta versão (v2):
# 1) server_val usa transform SEM augment (antes usava o train_tf com RandomCrop/Flip -> val_loss ruidoso)
# 2) Opção pra pular RANDOM e começar direto em probe:
#    - --skip_random  (remove "random" da lista de policies)
#    - --start_policy probe_high  (ordena para começar por uma policy específica)
# 3) Policy extra: probe_mid (evita extremos, seleciona no meio do ranking)
# 4) Logs melhores: também imprime o número total de clientes poisoned e percentuais selecionados.
#
# Run exemplos:
#   (padrão)   python ablation_probe_selection_cifar10_v2.py
#   (3 seeds)  python ablation_probe_selection_cifar10_v2.py --seeds 0 1 2
#   (pular random e começar em probe) python ablation_probe_selection_cifar10_v2.py --skip_random --start_policy probe_high
#   (ataque)   python ablation_probe_selection_cifar10_v2.py --poison_frac 0.5 --flip_frac 0.7 --skip_random --start_policy probe_low
#
# Requisitos: torch, torchvision, numpy

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# -------------------------
# Reprodutibilidade
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Modelo simples (CIFAR-10)
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.c1(x)))   # 32x16x16
        x = self.pool(F.relu(self.c2(x)))   # 64x8x8
        x = self.pool(F.relu(self.c3(x)))   # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# Dataset wrapper p/ label flip determinístico (opcional)
# -------------------------
class LabelFlipView(Dataset):
    """
    View sobre subset de um dataset base, com label flipping determinístico em alguns índices.
    flip_map: dict {global_index_in_base_dataset: new_label}
    """
    def __init__(self, base_ds: Dataset, indices: List[int], flip_map: Dict[int, int]):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.flip_map = flip_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        x, y = self.base_ds[base_idx]
        if base_idx in self.flip_map:
            y = self.flip_map[base_idx]
        return x, y


# -------------------------
# Partição Dirichlet (não-IID)
# -------------------------
def dirichlet_partition(targets: np.ndarray, n_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    n_classes = int(targets.max()) + 1
    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        proportions = np.maximum(proportions, 1e-8)
        proportions = proportions / proportions.sum()

        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for i in range(n_clients):
            client_indices[i].extend(splits[i].tolist())

    for i in range(n_clients):
        rng.shuffle(client_indices[i])
    return client_indices


# -------------------------
# Métricas / avaliação
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def probing_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int) -> float:
    model.eval()
    losses = []
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        losses.append(loss.item())
    if not losses:
        return float("nan")
    return float(np.mean(losses))


# -------------------------
# Treino local e agregação
# -------------------------
def train_local(
    global_model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_steps: int = None,
) -> Dict[str, torch.Tensor]:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def fedavg_aggregate(
    global_sd: Dict[str, torch.Tensor],
    locals_sds: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    new_sd = {}
    for k in global_sd.keys():
        acc = None
        for sd_i, w in zip(locals_sds, weights):
            term = sd_i[k] * (w / total)
            acc = term if acc is None else (acc + term)
        new_sd[k] = acc
    return new_sd


# -------------------------
# Policies
# -------------------------
def select_clients(policy: str, K: int, probe_losses: np.ndarray, rng: np.random.Generator) -> List[int]:
    n = len(probe_losses)

    if policy == "random":
        return rng.choice(n, size=K, replace=False).tolist()

    order_asc = np.argsort(probe_losses)       # menor -> maior
    order_desc = order_asc[::-1]               # maior -> menor

    if policy == "probe_high":
        return order_desc[:K].tolist()

    if policy == "probe_low":
        return order_asc[:K].tolist()

    if policy == "probe_mid":
        # evita extremos: pega do "meio" do ranking
        # Ex.: descarta bottom p% e top p%, pega K do meio (aprox).
        p = 0.2  # 20% extremos descartados
        lo = int(np.floor(p * n))
        hi = int(np.ceil((1.0 - p) * n))
        mid = order_asc[lo:hi]
        if len(mid) <= K:
            # fallback: se n pequeno
            return rng.choice(n, size=K, replace=False).tolist()
        # pega K do meio (poderia randomizar; aqui é determinístico)
        start = (len(mid) - K) // 2
        return mid[start:start + K].tolist()

    raise ValueError(f"policy desconhecida: {policy}")


# -------------------------
# Sumários finais
# -------------------------
def auc_trapz(y: List[float]) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.trapz(np.array(y), dx=1.0) / (len(y) - 1))


def time_to_threshold(y: List[float], thr: float) -> int:
    for i, v in enumerate(y, start=1):
        if v >= thr:
            return i
    return -1


def tail_std(y: List[float], last_n: int) -> float:
    if len(y) < 2:
        return 0.0
    tail = y[-last_n:] if len(y) >= last_n else y
    return float(np.std(tail))


# -------------------------
# Experimento
# -------------------------
@dataclass
class RunResult:
    policy: str
    seed: int
    val_loss: List[float]
    test_acc: List[float]
    probe_sel_mean: List[float]
    probe_sel_std: List[float]
    auc_test: float
    t60: int
    t65: int
    t70: int
    tail_std_acc: float


def build_balanced_val_indices(targets: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(targets)
    n_classes = int(targets.max()) + 1
    per_class = int((n * val_frac) / n_classes)
    val_indices = []
    for c in range(n_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        val_indices.extend(idx_c[:per_class].tolist())
    return np.array(val_indices, dtype=np.int64)


def run_one(
    policy: str,
    seed: int,
    device: str,
    rounds: int,
    n_clients: int,
    K: int,
    alpha: float,
    batch_size: int,
    local_epochs: int,
    local_lr: float,
    weight_decay: float,
    probe_batches: int,
    # opcional: ataque
    poison_frac: float,
    flip_frac: float,
    log_dir: str,
    # flip rule
    flip_rule: str,
) -> RunResult:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dois datasets para o MESMO CIFAR-10 train:
    # - um com augment (treino local / probing se você quiser)
    # - um sem augment (server_val)
    train_aug = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR10(root="./data", train=True, download=False, transform=eval_tf)

    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

    targets = np.array(train_aug.targets, dtype=np.int64)
    all_indices = np.arange(len(train_aug))

    # server_val separado (10% do train, balanceado)
    val_indices = build_balanced_val_indices(targets, val_frac=0.1, seed=seed)
    val_mask = np.zeros(len(train_aug), dtype=bool)
    val_mask[val_indices] = True
    pool_indices = all_indices[~val_mask]

    # Particiona o pool entre clientes (não-IID)
    pool_targets = targets[pool_indices]
    client_splits_local = dirichlet_partition(pool_targets, n_clients=n_clients, alpha=alpha, seed=seed)

    # converte para índices do dataset base
    client_indices = [pool_indices[np.array(idxs, dtype=np.int64)].tolist() for idxs in client_splits_local]

    # Attack (label flip determinístico)
    flip_map: Dict[int, int] = {}
    poisoned_clients = set()
    if poison_frac > 0.0 and flip_frac > 0.0:
        n_poison = int(round(poison_frac * n_clients))
        poisoned_clients = set(rng.choice(n_clients, size=n_poison, replace=False).tolist())

        for cid in poisoned_clients:
            idxs = client_indices[cid]
            m = int(round(flip_frac * len(idxs)))
            if m <= 0:
                continue
            chosen = rng.choice(len(idxs), size=m, replace=False)
            for j in chosen:
                base_idx = idxs[j]
                y = int(targets[base_idx])
                if flip_rule == "plus1":
                    newy = (y + 1) % 10
                elif flip_rule == "inv":
                    newy = 9 - y
                elif flip_rule == "to0":
                    newy = 0
                else:
                    raise ValueError(f"flip_rule desconhecida: {flip_rule}")
                flip_map[base_idx] = int(newy)

    # Loaders server (VAL sem augment!)
    val_loader = DataLoader(Subset(train_eval, val_indices.tolist()), batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    # Loaders por cliente:
    # - treino local: usa train_aug (com augment) para simular treino realista
    # - probing: por padrão usa train_eval (sem augment) para o probe ficar mais estável
    client_loaders = []
    client_probe_loaders = []
    client_sizes = []

    for cid in range(n_clients):
        idxs = client_indices[cid]

        if cid in poisoned_clients:
            ds_train = LabelFlipView(train_aug, idxs, flip_map)
            ds_probe = LabelFlipView(train_eval, idxs, flip_map)
        else:
            ds_train = Subset(train_aug, idxs)
            ds_probe = Subset(train_eval, idxs)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        probe_loader = DataLoader(ds_probe, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

        client_loaders.append(train_loader)
        client_probe_loaders.append(probe_loader)
        client_sizes.append(len(idxs))

    # Modelo global
    model = SmallCNN().to(device)
    global_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Logs
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{policy}_seed{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    val_loss_hist, test_acc_hist = [], []
    probe_sel_mean_hist, probe_sel_std_hist = [], []

    poisoned_total = len(poisoned_clients)

    for r in range(1, rounds + 1):
        model.load_state_dict(global_sd, strict=True)
        model.to(device)

        # 1) avaliar no server (val/test)
        vloss, vacc = evaluate(model, val_loader, device=device)
        tloss, tacc = evaluate(model, test_loader, device=device)

        # 2) probing em todos os clientes (antes da seleção)
        probe_losses = np.zeros(n_clients, dtype=np.float32)
        for cid in range(n_clients):
            pl = probing_loss(model, client_probe_loaders[cid], device=device, max_batches=probe_batches)
            probe_losses[cid] = pl

        # 3) seleção
        selected = select_clients(policy, K=K, probe_losses=probe_losses, rng=rng)

        sel_probe = probe_losses[selected]
        sel_probe_mean = float(np.mean(sel_probe))
        sel_probe_std = float(np.std(sel_probe))

        # 4) treino local
        locals_sds = []
        weights = []
        for cid in selected:
            local_sd = train_local(
                model, client_loaders[cid], device=device,
                epochs=local_epochs, lr=local_lr, weight_decay=weight_decay
            )
            locals_sds.append(local_sd)
            weights.append(client_sizes[cid])

        # 5) FedAvg
        global_sd = fedavg_aggregate(global_sd, locals_sds, weights)

        # 6) guardar métricas
        val_loss_hist.append(float(vloss))
        test_acc_hist.append(float(tacc))
        probe_sel_mean_hist.append(sel_probe_mean)
        probe_sel_std_hist.append(sel_probe_std)

        # 7) logs
        top5 = np.argsort(-probe_losses)[:5].tolist()
        low5 = np.argsort(probe_losses)[:5].tolist()
        poisoned_in_sel = int(sum(1 for cid in selected if cid in poisoned_clients)) if poisoned_total > 0 else 0
        poison_pct_sel = (100.0 * poisoned_in_sel / K) if K > 0 else 0.0

        log_obj = {
            "round": r,
            "policy": policy,
            "seed": seed,
            "val_loss": float(vloss),
            "val_acc": float(vacc),
            "test_loss": float(tloss),
            "test_acc": float(tacc),
            "sel": selected,
            "sel_probe_mean": sel_probe_mean,
            "sel_probe_std": sel_probe_std,
            "probe_top5_clients": top5,
            "probe_low5_clients": low5,
            "poisoned_clients_total": poisoned_total,
            "poisoned_in_selected": poisoned_in_sel,
            "flip_rule": flip_rule,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_obj) + "\n")

        print(
            f"[{policy} | seed {seed}] round {r:4d}/{rounds} | "
            f"val_loss={vloss:.4f} | test_acc={tacc*100:6.2f}% | "
            f"probe_sel(mean±std)={sel_probe_mean:.4f}±{sel_probe_std:.4f} | "
            f"poison_sel={poisoned_in_sel}/{K} ({poison_pct_sel:4.1f}%)"
        )

    # Sumários
    auc = auc_trapz(test_acc_hist)
    t60 = time_to_threshold(test_acc_hist, 0.60)
    t65 = time_to_threshold(test_acc_hist, 0.65)
    t70 = time_to_threshold(test_acc_hist, 0.70)
    tail = tail_std(test_acc_hist, last_n=min(30, rounds))

    summary = {
        "policy": policy,
        "seed": seed,
        "AUC_test_acc": auc,
        "t60": t60,
        "t65": t65,
        "t70": t70,
        "tail_std_acc": tail,
        "final_test_acc": float(test_acc_hist[-1]) if test_acc_hist else None,
        "final_val_loss": float(val_loss_hist[-1]) if val_loss_hist else None,
        "poisoned_clients_total": poisoned_total,
        "poison_frac": poison_frac,
        "flip_frac": flip_frac,
        "flip_rule": flip_rule,
    }
    with open(os.path.join(log_dir, f"summary_{policy}_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return RunResult(
        policy=policy,
        seed=seed,
        val_loss=val_loss_hist,
        test_acc=test_acc_hist,
        probe_sel_mean=probe_sel_mean_hist,
        probe_sel_std=probe_sel_std_hist,
        auc_test=auc,
        t60=t60, t65=t65, t70=t70,
        tail_std_acc=tail,
    )


def order_policies(policies: List[str], start_policy: str) -> List[str]:
    if start_policy is None:
        return policies
    if start_policy not in policies:
        return policies
    # move start_policy pro começo mantendo ordem relativa do resto
    return [start_policy] + [p for p in policies if p != start_policy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n_clients", type=int, default=50)
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha (menor => mais não-IID)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--local_lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--probe_batches", type=int, default=2, help="n batches para probing por cliente")
    ap.add_argument("--policies", nargs="+", default=["random", "probe_high", "probe_low", "probe_mid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    # ataque opcional
    ap.add_argument("--poison_frac", type=float, default=0.0)
    ap.add_argument("--flip_frac", type=float, default=0.0)
    ap.add_argument("--flip_rule", type=str, default="plus1", choices=["plus1", "inv", "to0"])

    # logs
    ap.add_argument("--log_dir", type=str, default="./logs_probe_ablation_v2")

    # opções novas (pular random / começar em probe)
    ap.add_argument("--skip_random", action="store_true", help="remove 'random' das policies")
    ap.add_argument("--start_policy", type=str, default=None,
                    help="coloca esta policy primeiro (ex: probe_high, probe_low, probe_mid)")

    args = ap.parse_args()

    policies = list(args.policies)
    if args.skip_random and "random" in policies:
        policies = [p for p in policies if p != "random"]
    policies = order_policies(policies, args.start_policy)

    print(f"Device: {args.device}")
    print(f"Policies: {policies} | Seeds: {args.seeds}")
    print(f"Attack: poison_frac={args.poison_frac}, flip_frac={args.flip_frac}, flip_rule={args.flip_rule}")
    os.makedirs(args.log_dir, exist_ok=True)

    all_results: List[RunResult] = []

    # Ordem recomendada: por seed, roda todas as policies (melhor p/ comparar cedo)
    for seed in args.seeds:
        for policy in policies:
            res = run_one(
                policy=policy,
                seed=seed,
                device=args.device,
                rounds=args.rounds,
                n_clients=args.n_clients,
                K=args.K,
                alpha=args.alpha,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                local_lr=args.local_lr,
                weight_decay=args.weight_decay,
                probe_batches=args.probe_batches,
                poison_frac=args.poison_frac,
                flip_frac=args.flip_frac,
                log_dir=args.log_dir,
                flip_rule=args.flip_rule,
            )
            all_results.append(res)

    # Tabela final agregada
    print("\n================ FINAL (média por policy) ================\n")
    by_pol: Dict[str, List[RunResult]] = {}
    for r in all_results:
        by_pol.setdefault(r.policy, []).append(r)

    def mean_std(vals):
        return float(np.mean(vals)), float(np.std(vals))

    for pol, rs in by_pol.items():
        aucs = [x.auc_test for x in rs]
        finals = [x.test_acc[-1] for x in rs]
        t70s = [x.t70 for x in rs if x.t70 != -1]
        tail = [x.tail_std_acc for x in rs]

        auc_m, auc_s = mean_std(aucs)
        fin_m, fin_s = mean_std(finals)
        tail_m, tail_s = mean_std(tail)

        if len(t70s) > 0:
            t70_m, t70_s = mean_std(t70s)
            t70_str = f"{t70_m:.1f}±{t70_s:.1f}"
        else:
            t70_str = "NA"

        print(
            f"{pol:10s} | AUC(test_acc)={auc_m:.4f}±{auc_s:.4f} | "
            f"final_acc={fin_m*100:6.2f}%±{fin_s*100:5.2f}% | "
            f"t70={t70_str} | tail_std_acc={tail_m:.4f}±{tail_s:.4f}"
        )

    print(f"\nLogs em: {args.log_dir}")
    print(" - round-by-round: log_<policy>_seed<k>.jsonl")
    print(" - summary: summary_<policy>_seed<k>.json")


if __name__ == "__main__":
    main()
