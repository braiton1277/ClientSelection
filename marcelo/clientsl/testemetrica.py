# fl_fedavg_gp_score_topk_cifar10.py
# ============================================================
# Federated Learning (CIFAR-10) com seleção Top-K por
# "Gradient Projection (GP) score" EXATAMENTE como na fórmula:
#
#   c_i^t = ( ∇F(w_i^t) · ∇F(w^{t-1}) ) / || ∇F(w^{t-1}) ||
#
# Onde:
# - ∇F(w^{t-1}) = gradiente "global" (referência) calculado no servidor
#   em um conjunto server_ref (holdout do TRAIN), usando o modelo global atual w.
# - ∇F(w_i^t)   = gradiente do cliente i calculado nos dados do cliente
#   usando o MESMO modelo global atual (sem atualizar peso).
#
# Seleção:
# - Seleciona Top-K com MAIOR c_i^t (mais alinhado/forte na direção global).
#
# Treino FL:
# - Depois de selecionar, executa treino local real (SGD) só nos selecionados,
#   agrega via FedAvg (média dos deltas de parâmetros).
#
# Randomização opcional:
# - Warmup aleatório nas primeiras N rodadas: --random_warmup_rounds
# - ε-greedy: com probabilidade p escolhe aleatório: --random_prob
#
# Ataque:
# - Label flip determinístico por amostra em fração de clientes (atacantes),
#   com opção targeted.
#
# Como rodar:
#   python fl_fedavg_gp_score_topk_cifar10.py
#   python fl_fedavg_gp_score_topk_cifar10.py --rounds 50 --random_warmup_rounds 10
#   python fl_fedavg_gp_score_topk_cifar10.py --random_prob 0.1
#   python fl_fedavg_gp_score_topk_cifar10.py --enable_label_flip 1 --flip_fraction 0.4 --flip_rate 1.0 --targeted_flip 1
# ============================================================

import argparse
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ----------------------------
# Config (defaults)
# ----------------------------
@dataclass
class Cfg:
    seed: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FL
    rounds: int = 400
    n_clients: int = 50
    k_select: int = 15
    dirichlet_alpha: float = 0.3
    min_client_size: int = 200

    # Local train (FedAvg)
    client_batch: int = 64
    local_lr: float = 0.01
    local_steps: int = 20
    local_momentum: float = 0.9

    # Server ref set (para gradiente global ∇F(w^{t-1}))
    server_ref_per_class: int = 200       # 200*10=2000 amostras de referência
    server_ref_batch: int = 256
    server_ref_grad_batches: int = 10     # quantos batches para estimar ∇F global

    # Cliente grad (para ∇F(w_i^t))
    client_grad_batch: int = 128
    client_grad_batches: int = 2          # quantos batches por cliente para estimar grad

    # Server test eval
    test_batch: int = 256

    # Logging
    print_rank_every: int = 5

    # Randomization
    random_warmup_rounds: int = 25         # primeiras N rodadas aleatórias
    random_prob: float = 0.0              # ε-greedy: prob de seleção aleatória

    # Label flip (atacantes)
    enable_label_flip: bool = True
    flip_fraction: float = 0.4
    flip_rate: float = 0.6
    targeted_flip: bool = True


CFG = Cfg()


# ----------------------------
# Utils: seed
# ----------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = CFG.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ----------------------------
# Model (small + fast)
# ----------------------------
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)  # 32->16->8->4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_steps(model: nn.Module, loader: DataLoader, lr: float, steps_cap: int, momentum: float = 0.9):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum))

    it = iter(loader)
    for _ in range(int(steps_cap)):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(CFG.device), y.to(CFG.device)
        loss = F.cross_entropy(model(x), y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def flat_grads(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads)


def load_flat_params_(model: nn.Module, flat: torch.Tensor):
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[off:off + n].view_as(p))
        off += n


def local_train_delta(global_model: nn.Module, loader: DataLoader) -> torch.Tensor:
    m = copy.deepcopy(global_model).to(CFG.device)
    w0 = flat_params(m).clone()

    train_steps(
        model=m,
        loader=loader,
        lr=CFG.local_lr,
        steps_cap=CFG.local_steps,
        momentum=CFG.local_momentum,
    )

    w1 = flat_params(m)
    return (w1 - w0).detach()


def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor]):
    w = flat_params(model).clone()
    avg_dw = torch.stack(deltas, dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


# ----------------------------
# Label flip determinístico por amostra
# ----------------------------
def build_targeted_map(num_classes: int = 10) -> Dict[int, int]:
    # targeted simples: i -> i+1
    return {i: (i + 1) % num_classes for i in range(num_classes)}


class FlippedSubset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        indices: List[int],
        enable_flip: bool,
        flip_rate: float,
        targeted: bool,
        seed: int,
        num_classes: int = 10,
    ):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.enable_flip = bool(enable_flip)
        self.flip_rate = float(flip_rate)
        self.targeted = bool(targeted)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.tmap = build_targeted_map(num_classes)

    def __len__(self):
        return len(self.indices)

    def _should_flip(self, global_idx: int) -> bool:
        if (not self.enable_flip) or (self.flip_rate <= 0):
            return False
        rng = np.random.default_rng(self.seed + int(global_idx) * 9973)
        return (rng.random() < self.flip_rate)

    def _flip_label(self, y: int, global_idx: int) -> int:
        y = int(y)
        if self.targeted:
            return int(self.tmap[y])
        rng = np.random.default_rng(self.seed + int(global_idx) * 7919 + 123)
        choices = list(range(self.num_classes))
        choices.remove(y)
        return int(rng.choice(choices))

    def __getitem__(self, i: int):
        gidx = int(self.indices[i])
        x, y = self.base_ds[gidx]
        if self._should_flip(gidx):
            y = self._flip_label(y, gidx)
        return x, y


# ----------------------------
# Dirichlet split (por classe)
# ----------------------------
def dirichlet_split(labels: np.ndarray, n_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max() + 1)
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    clients = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = idx_by_class[c]
        props = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (props * len(idx_c)).astype(int)

        diff = len(idx_c) - counts.sum()
        if diff > 0:
            counts[rng.integers(0, n_clients, size=diff)] += 1
        elif diff < 0:
            for _ in range(-diff):
                j = int(rng.integers(0, n_clients))
                if counts[j] > 0:
                    counts[j] -= 1

        start = 0
        for j in range(n_clients):
            take = int(counts[j])
            if take > 0:
                clients[j].extend(idx_c[start:start + take].tolist())
                start += take

    for j in range(n_clients):
        rng.shuffle(clients[j])
    return clients


def enforce_min_size(client_indices: List[List[int]], min_size: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed + 999)
    sizes = np.array([len(x) for x in client_indices], dtype=int)
    small = np.where(sizes < min_size)[0].tolist()
    if not small:
        return client_indices

    big = np.where(sizes > min_size)[0].tolist()
    if not big:
        return client_indices

    for s in small:
        need = min_size - len(client_indices[s])
        while need > 0 and big:
            b = int(rng.choice(big))
            if len(client_indices[b]) <= min_size:
                big.remove(b)
                continue
            moved = client_indices[b].pop()
            client_indices[s].append(moved)
            need -= 1
    return client_indices


# ----------------------------
# Server ref set balanceado (holdout do TRAIN)
# ----------------------------
def make_server_ref_balanced(ds, per_class: int, n_classes: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(ds)):
        _, y = ds[idx]
        label_to_idxs[int(y)].append(idx)

    ref = []
    for y in range(n_classes):
        idxs = label_to_idxs[y]
        rng.shuffle(idxs)
        ref.extend(idxs[:per_class])

    rng.shuffle(ref)
    return ref


# ----------------------------
# Gradientes: global (servidor) e por cliente
# ----------------------------
def zero_grads_(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def compute_reference_grad_server(model: nn.Module, server_ref_loader: DataLoader, max_batches: int) -> torch.Tensor:
    """
    Retorna ∇F(w^{t-1}) estimado no servidor (server_ref).
    """
    model.train()
    zero_grads_(model)

    for b, (x, y) in enumerate(server_ref_loader):
        if b >= max_batches:
            break
        x, y = x.to(CFG.device), y.to(CFG.device)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    gref = flat_grads(model).detach()
    zero_grads_(model)
    return gref


def compute_client_grad(model: nn.Module, client_grad_loader: DataLoader, max_batches: int) -> torch.Tensor:
    """
    Retorna ∇F(w_i^t) estimado no cliente i (em batches dos dados do cliente),
    usando o MESMO modelo global (sem atualizar pesos).
    """
    model.train()
    zero_grads_(model)

    for b, (x, y) in enumerate(client_grad_loader):
        if b >= max_batches:
            break
        x, y = x.to(CFG.device), y.to(CFG.device)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    gi = flat_grads(model).detach()
    zero_grads_(model)
    return gi


def gp_score_exact(gi: torch.Tensor, gref: torch.Tensor) -> float:
    """
    IMPLEMENTA EXATAMENTE:
        c_i^t = (gi · gref) / ||gref||
    """
    denom = float(gref.norm().item()) + 1e-12
    num = float(torch.dot(gi, gref).item())
    return num / denom


def select_topk_by_gp_score(
    model: nn.Module,
    server_ref_loader: DataLoader,
    client_grad_loaders: List[DataLoader],
) -> Tuple[List[int], List[Tuple[int, str, float]]]:
    """
    Calcula gref no servidor, depois c_i^t para cada cliente.
    Seleciona Top-K por MAIOR score (desc).
    """
    gref = compute_reference_grad_server(model, server_ref_loader, max_batches=CFG.server_ref_grad_batches)

    scores = []
    for cid in range(CFG.n_clients):
        gi = compute_client_grad(model, client_grad_loaders[cid], max_batches=CFG.client_grad_batches)
        ci = gp_score_exact(gi, gref)
        scores.append(ci)

    order = np.argsort(-np.array(scores, dtype=np.float32))  # DESC
    K = min(CFG.k_select, CFG.n_clients)
    selected = order[:K].tolist()
    topk_info = [(cid, "", float(scores[cid])) for cid in selected]
    return selected, topk_info


# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--rounds", type=int, default=CFG.rounds)
    p.add_argument("--n_clients", type=int, default=CFG.n_clients)
    p.add_argument("--k_select", type=int, default=CFG.k_select)
    p.add_argument("--dirichlet_alpha", type=float, default=CFG.dirichlet_alpha)
    p.add_argument("--min_client_size", type=int, default=CFG.min_client_size)

    p.add_argument("--client_batch", type=int, default=CFG.client_batch)
    p.add_argument("--local_lr", type=float, default=CFG.local_lr)
    p.add_argument("--local_steps", type=int, default=CFG.local_steps)
    p.add_argument("--local_momentum", type=float, default=CFG.local_momentum)

    p.add_argument("--server_ref_per_class", type=int, default=CFG.server_ref_per_class)
    p.add_argument("--server_ref_batch", type=int, default=CFG.server_ref_batch)
    p.add_argument("--server_ref_grad_batches", type=int, default=CFG.server_ref_grad_batches)

    p.add_argument("--client_grad_batch", type=int, default=CFG.client_grad_batch)
    p.add_argument("--client_grad_batches", type=int, default=CFG.client_grad_batches)

    p.add_argument("--print_rank_every", type=int, default=CFG.print_rank_every)

    p.add_argument("--random_warmup_rounds", type=int, default=CFG.random_warmup_rounds)
    p.add_argument("--random_prob", type=float, default=CFG.random_prob)

    p.add_argument("--enable_label_flip", type=int, default=1 if CFG.enable_label_flip else 0)
    p.add_argument("--flip_fraction", type=float, default=CFG.flip_fraction)
    p.add_argument("--flip_rate", type=float, default=CFG.flip_rate)
    p.add_argument("--targeted_flip", type=int, default=1 if CFG.targeted_flip else 0)
    return p.parse_args()


def apply_args(args):
    CFG.seed = int(args.seed)
    CFG.rounds = int(args.rounds)
    CFG.n_clients = int(args.n_clients)
    CFG.k_select = int(args.k_select)
    CFG.dirichlet_alpha = float(args.dirichlet_alpha)
    CFG.min_client_size = int(args.min_client_size)

    CFG.client_batch = int(args.client_batch)
    CFG.local_lr = float(args.local_lr)
    CFG.local_steps = int(args.local_steps)
    CFG.local_momentum = float(args.local_momentum)

    CFG.server_ref_per_class = int(args.server_ref_per_class)
    CFG.server_ref_batch = int(args.server_ref_batch)
    CFG.server_ref_grad_batches = int(args.server_ref_grad_batches)

    CFG.client_grad_batch = int(args.client_grad_batch)
    CFG.client_grad_batches = int(args.client_grad_batches)

    CFG.print_rank_every = int(args.print_rank_every)

    CFG.random_warmup_rounds = int(args.random_warmup_rounds)
    CFG.random_prob = float(args.random_prob)

    CFG.enable_label_flip = bool(int(args.enable_label_flip) != 0)
    CFG.flip_fraction = float(args.flip_fraction)
    CFG.flip_rate = float(args.flip_rate)
    CFG.targeted_flip = bool(int(args.targeted_flip) != 0)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    apply_args(args)
    seed_all(CFG.seed)

    print(f"[Device] {CFG.device} | seed={CFG.seed}")
    print(f"[FL] rounds={CFG.rounds} | n_clients={CFG.n_clients} | K={CFG.k_select} | alpha={CFG.dirichlet_alpha}")
    print(f"[GP score] server_ref_batches={CFG.server_ref_grad_batches} | client_grad_batches={CFG.client_grad_batches}")
    print(f"[Random] warmup_rounds={CFG.random_warmup_rounds} | random_prob={CFG.random_prob}")
    print(f"[LocalTrain] steps={CFG.local_steps} | lr={CFG.local_lr} | batch={CFG.client_batch}\n")

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Train com AUG (treino local) e Train sem AUG (para grad/score mais estável)
    train_ds_aug  = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf_train)
    train_ds_eval = datasets.CIFAR10(root="./data", train=True, download=False, transform=tf_eval)
    test_ds       = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_eval)

    labels = np.array(train_ds_aug.targets, dtype=int)

    # --- Server ref balanced do TRAIN (holdout) ---
    server_ref_idxs = make_server_ref_balanced(train_ds_eval, per_class=CFG.server_ref_per_class, n_classes=10, seed=CFG.seed + 4242)
    server_ref_set = set(server_ref_idxs)

    # pool para clientes = train inteiro menos server_ref
    all_idx = np.arange(len(train_ds_aug))
    pool_idx = [int(i) for i in all_idx if int(i) not in server_ref_set]
    rng = np.random.default_rng(CFG.seed)
    rng.shuffle(pool_idx)

    pool_labels = labels[pool_idx]

    # Dirichlet split no pool
    client_local = dirichlet_split(pool_labels, CFG.n_clients, CFG.dirichlet_alpha, CFG.seed)
    client_global = [[pool_idx[i] for i in cl] for cl in client_local]
    client_global = enforce_min_size(client_global, CFG.min_client_size, CFG.seed)

    # Escolhe atacantes
    n_attackers = int(round(CFG.n_clients * CFG.flip_fraction)) if CFG.enable_label_flip else 0
    attacker_set = set(rng.choice(CFG.n_clients, size=n_attackers, replace=False).tolist()) if n_attackers > 0 else set()

    if CFG.enable_label_flip:
        print(f"[LabelFlip ON] flip_fraction={CFG.flip_fraction} -> n_attackers={n_attackers} | flip_rate={CFG.flip_rate} | targeted={CFG.targeted_flip}")
        print(f"[Attackers] {sorted(list(attacker_set))}\n")
    else:
        print("[LabelFlip OFF]\n")

    # Loaders:
    # - train loader: AUG + (labels flipados se atacante)
    # - grad loader: EVAL (sem AUG) + (labels flipados se atacante) => usado pra ∇F(w_i^t)
    client_train_loaders: List[DataLoader] = []
    client_grad_loaders: List[DataLoader] = []
    client_sizes: List[int] = []

    for cid in range(CFG.n_clients):
        is_attacker = (cid in attacker_set)

        ds_train = FlippedSubset(
            base_ds=train_ds_aug,
            indices=client_global[cid],
            enable_flip=is_attacker and CFG.enable_label_flip,
            flip_rate=CFG.flip_rate,
            targeted=CFG.targeted_flip,
            seed=CFG.seed + 10000 + cid * 17,
            num_classes=10,
        )
        ds_grad = FlippedSubset(
            base_ds=train_ds_eval,
            indices=client_global[cid],
            enable_flip=is_attacker and CFG.enable_label_flip,
            flip_rate=CFG.flip_rate,
            targeted=CFG.targeted_flip,
            seed=CFG.seed + 10000 + cid * 17,
            num_classes=10,
        )

        client_sizes.append(len(ds_train))

        dl_train = DataLoader(
            ds_train,
            batch_size=CFG.client_batch,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
        )
        dl_grad = DataLoader(
            ds_grad,
            batch_size=CFG.client_grad_batch,
            shuffle=True,   # pega batches “aleatórios” por rodada
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
        )

        client_train_loaders.append(dl_train)
        client_grad_loaders.append(dl_grad)

    print(f"[Clients] avg_size={np.mean(client_sizes):.1f} | min={np.min(client_sizes)} | max={np.max(client_sizes)}")
    print(f"[ServerRef] size={len(server_ref_idxs)} (per_class={CFG.server_ref_per_class})\n")

    server_ref_loader = DataLoader(
        Subset(train_ds_eval, server_ref_idxs),
        batch_size=CFG.server_ref_batch,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(test_ds, batch_size=CFG.test_batch, shuffle=False, num_workers=0)

    model = SmallCNN().to(CFG.device)

    # Loop FL
    for t in range(1, CFG.rounds + 1):
        acc = eval_acc(model, test_loader)
        K = min(CFG.k_select, CFG.n_clients)

        # Decide modo (random warmup ou eps-greedy)
        do_random = False
        if CFG.random_warmup_rounds > 0 and t <= CFG.random_warmup_rounds:
            do_random = True
        else:
            if CFG.random_prob > 0.0 and (rng.random() < CFG.random_prob):
                do_random = True

        if do_random:
            selected = rng.choice(CFG.n_clients, size=K, replace=False).tolist()
            mode = "RANDOM"
            topk_info_tagged = [(cid, ("ATTACKER" if cid in attacker_set else "OK"), float("nan")) for cid in selected]
            attackers_in_sel = sum([1 for cid in selected if cid in attacker_set])
        else:
            selected, topk_info = select_topk_by_gp_score(model, server_ref_loader, client_grad_loaders)
            mode = "GP"

            # Tag attackers + scores
            topk_info_sorted = sorted(topk_info, key=lambda x: x[2], reverse=True)  # maior score primeiro
            topk_info_tagged = []
            attackers_in_sel = 0
            for (cid, _, scorev) in topk_info_sorted:
                tag = "ATTACKER" if cid in attacker_set else "OK"
                if tag == "ATTACKER":
                    attackers_in_sel += 1
                topk_info_tagged.append((cid, tag, scorev))

        # Treina local só nos selecionados e agrega
        deltas = []
        for cid in selected:
            dw = local_train_delta(model, client_train_loaders[cid])
            deltas.append(dw)
        apply_fedavg(model, deltas)

        if mode == "GP":
            best_score = topk_info_tagged[0][2]
            worst_score = topk_info_tagged[-1][2]
            print(
                f"[round {t:03d}] mode={mode} | TEST acc={acc*100:6.2f}% | "
                f"selected_attackers={attackers_in_sel}/{len(selected)} | "
                f"best_score={best_score:+.6f} worst_score(topK)={worst_score:+.6f}"
            )
        else:
            print(
                f"[round {t:03d}] mode={mode} | TEST acc={acc*100:6.2f}% | "
                f"selected_attackers={attackers_in_sel}/{len(selected)}"
            )

        if CFG.print_rank_every > 0 and (t % CFG.print_rank_every == 0):
            if mode == "GP":
                print(f"  [Top-{len(selected)} by GP score] cid | tag | score")
                for (cid, tag, scorev) in topk_info_tagged:
                    print(f"    {cid:02d} | {tag:8s} | score={scorev:+.6f}")
            else:
                print(f"  [Selected-{len(selected)} RANDOM] cid | tag")
                for (cid, tag, _) in topk_info_tagged:
                    print(f"    {cid:02d} | {tag:8s}")
            print("")

    print("Done.")


if __name__ == "__main__":
    main()

