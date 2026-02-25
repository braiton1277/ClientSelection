# fl_resnet_randomsearch_dirichlet_pure.py
# ------------------------------------------------------------
# Federated Learning "na mão" (sem Flower):
# - 50 clientes
# - Split Dirichlet(alpha) por classe no CIFAR-10
# - Seleção aleatória explícita de K clientes por rodada
# - FedAvg
# - ResNet-18 adaptada para CIFAR-10
# - Random search de hiperparâmetros (print only)
#
# Reqs:
#   pip install torch torchvision numpy
#
# Rodar:
#   python fl_resnet_randomsearch_dirichlet_pure.py --trials 8 --rounds 60 --k 15 --alpha 0.3 --seeds 111 2048
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# -----------------------------
# Reprodutibilidade
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# ResNet-18 adaptada CIFAR-10
# (conv3x3 stride1, sem maxpool)
# -----------------------------
def make_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------
# Dirichlet split por classe
# -----------------------------
def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = int(targets.max() + 1)

    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_c = idx_by_class[c]
            if len(idx_c) == 0:
                continue
            props = rng.dirichlet(alpha=np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)
            splits = np.split(idx_c, cuts[:-1])
            for i in range(num_clients):
                client_indices[i].extend(splits[i].tolist())

        sizes = np.array([len(x) for x in client_indices])
        if sizes.min() >= min_size:
            break

    return [np.array(rng.permutation(x), dtype=np.int64) for x in client_indices]


# -----------------------------
# Eval / Train
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def local_train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    grad_clip: float,
    scheduler: str,          # "none" | "cosine" | "step"
    scheduler_param: float,  # T_max (cosine) ou step_size (step)
) -> None:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    use_nesterov = (momentum > 0.0)

    opt = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=use_nesterov,
        )






    sch = None
    if scheduler == "cosine":
        T_max = max(int(scheduler_param), 1)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
    elif scheduler == "step":
        step_size = max(int(scheduler_param), 1)
        sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.1)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        if sch is not None:
            sch.step()


# -----------------------------
# FedAvg de state_dict (inclui buffers float)
# -----------------------------
def fedavg_state_dict(
    states: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}

    keys = states[0].keys()
    for k in keys:
        t0 = states[0][k]
        # se não for float (ex: num_batches_tracked), copia do primeiro
        if not torch.is_floating_point(t0):
            out[k] = t0.clone()
            continue

        acc = torch.zeros_like(t0)
        for st, w in zip(states, weights):
            acc += st[k] * (float(w) / total)
        out[k] = acc
    return out


# -----------------------------
# Random search: hiperparâmetros
# -----------------------------
@dataclass
class HParams:
    batch_size: int
    local_epochs: int
    lr: float
    momentum: float
    weight_decay: float
    grad_clip: float
    scheduler: str
    scheduler_param: float


def sample_hparams(rng: random.Random) -> HParams:
    batch_size = rng.choice([32, 64, 128])
    local_epochs = rng.choice([1, 2, 3, 5])

    lr = rng.choice([0.005, 0.01, 0.02, 0.05, 0.1])
    momentum = rng.choice([0.0, 0.8, 0.9, 0.95])
    weight_decay = rng.choice([0.0, 1e-4, 5e-4, 1e-3])
    grad_clip = rng.choice([0.0, 1.0, 5.0])

    scheduler = rng.choice(["none", "cosine", "step"])
    if scheduler == "cosine":
        scheduler_param = rng.choice([local_epochs, local_epochs * 2, 10])
    elif scheduler == "step":
        scheduler_param = rng.choice([1, 2, 3])
    else:
        scheduler_param = 0.0

    return HParams(
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        scheduler=scheduler,
        scheduler_param=float(scheduler_param),
    )


# -----------------------------
# Loop FL (sem Flower)
# -----------------------------
def run_fl(
    trainset,
    test_loader,
    client_splits: List[np.ndarray],
    num_clients: int,
    k: int,
    rounds: int,
    seed: int,
    hp: HParams,
    device: torch.device,
    num_workers: int,
) -> Tuple[List[float], float, float]:
    # modelo global
    global_model = make_resnet18_cifar10().to(device)
    global_state = copy.deepcopy(global_model.state_dict())

    # seleção aleatória reprodutível por rodada
    sel_rng = random.Random(seed)

    acc_curve: List[float] = []

    for r in range(1, rounds + 1):
        chosen = sel_rng.sample(range(num_clients), k=min(k, num_clients))

        local_states = []
        local_sizes = []

        for cid in chosen:
            idx = client_splits[cid]
            subset = Subset(trainset, idx.tolist())
            loader = DataLoader(
                subset,
                batch_size=hp.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            # cliente treina a partir do global
            model = make_resnet18_cifar10().to(device)
            model.load_state_dict(global_state, strict=True)

            local_train(
                model=model,
                loader=loader,
                device=device,
                epochs=hp.local_epochs,
                lr=hp.lr,
                momentum=hp.momentum,
                weight_decay=hp.weight_decay,
                grad_clip=hp.grad_clip,
                scheduler=hp.scheduler,
                scheduler_param=hp.scheduler_param,
            )

            local_states.append(copy.deepcopy(model.state_dict()))
            local_sizes.append(len(subset))

        # agrega
        global_state = fedavg_state_dict(local_states, local_sizes)
        global_model.load_state_dict(global_state, strict=True)

        # avalia central
        loss, acc = evaluate(global_model, test_loader, device)
        acc_curve.append(acc)

        print(f"round {r:03d}/{rounds} | loss={loss:.4f} acc={acc:.4f}")

    tail = acc_curve[-5:] if len(acc_curve) >= 5 else acc_curve
    score = float(np.mean(tail)) if len(tail) else 0.0
    best_acc = float(max(acc_curve)) if acc_curve else 0.0
    return acc_curve, score, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=50)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[111])
    ap.add_argument("--search_seed", type=int, default=999)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    search_rng = random.Random(args.search_seed)

    best = None
    best_score = -1.0

    for seed in args.seeds:
        set_all_seeds(seed)

        y = np.array(trainset.targets, dtype=np.int64)
        client_splits = dirichlet_partition(
            targets=y,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=seed,
            min_size=10,
        )

        for t in range(args.trials):
            hp = sample_hparams(search_rng)
            print("\n============================================================")
            print(f"[seed={seed}] trial {t+1}/{args.trials} | hparams={hp}")
            print("============================================================")

            _, score, best_acc = run_fl(
                trainset=trainset,
                test_loader=test_loader,
                client_splits=client_splits,
                num_clients=args.num_clients,
                k=args.k,
                rounds=args.rounds,
                seed=seed,
                hp=hp,
                device=device,
                num_workers=args.num_workers,
            )

            print(f"\nTRIAL DONE | score_last5_mean_acc={score:.4f} | best_acc={best_acc:.4f}\n")

            if score > best_score:
                best_score = score
                best = {"seed": seed, "hparams": hp, "score_last5_mean_acc": score, "best_acc": best_acc}
                print(">>> NEW BEST <<<")
                print(best)

    print("\n==================== BEST OVERALL ====================")
    print(best)
    print("======================================================\n")


if __name__ == "__main__":
    main()

