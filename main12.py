##testes de metricas do coseno
import copy
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------
# Reprodutibilidade TOTAL
# ----------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# (opcional) deixa CUDA mais determinístico (pode deixar um pouco mais lento)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_worker(worker_id: int):
    # garante reprodutibilidade dos workers do DataLoader
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g_dl = torch.Generator()
g_dl.manual_seed(SEED)


# ----------------------------
# Modelo simples (MLP)
# ----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ----------------------------
# Utils: flatten params/grads
# ----------------------------
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
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm() + eps)).item()


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, max_batches: int = 10) -> float:
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


def server_reference_grad(model: nn.Module, val_loader: DataLoader, batches: int = 10) -> torch.Tensor:
    """
    g_ref = ∇ L_val(w) no servidor (apontando para AUMENTAR a loss).
    Direção de descida é -g_ref.
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


def local_train_delta(global_model: nn.Module, loader: DataLoader, lr: float = 0.05, steps: int = 30) -> torch.Tensor:
    """
    Treina localmente alguns passos e devolve Δw = w_local - w_global.
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


# ----------------------------
# Split não-IID em 5 clientes por rótulo (pares de dígitos)
# ----------------------------
def make_5_clients_noniid_indices(train_ds) -> List[List[int]]:
    # c0={0,1}, c1={2,3}, c2={4,5}, c3={6,7}, c4={8,9}
    groups = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]

    label_to_idxs: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx in range(len(train_ds)):
        _, y = train_ds[idx]
        label_to_idxs[int(y)].append(idx)

    for y in range(10):
        random.shuffle(label_to_idxs[y])

    clients = []
    for gs in groups:
        idxs = []
        for y in gs:
            idxs.extend(label_to_idxs[y])
        random.shuffle(idxs)
        clients.append(idxs)

    return clients


# ----------------------------
# Val do servidor balanceado (IID-ish)
# ----------------------------
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


# ----------------------------
# Experimento
# ----------------------------
def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    # Validation do servidor (balanceado) -> 200*10=2000
    val_idxs = make_server_val_balanced(test_ds, per_class=200)
    val_loader = DataLoader(
        Subset(test_ds, val_idxs),
        batch_size=256,
        shuffle=True,
        generator=g_dl,
        worker_init_fn=seed_worker,
        num_workers=0,
    )

    # 5 clientes não-IID
    client_idxs = make_5_clients_noniid_indices(train_ds)

    client_loaders = []
    for idxs in client_idxs:
        idxs = idxs[:4000]  # pra rodar rápido
        client_loaders.append(
            DataLoader(
                Subset(train_ds, idxs),
                batch_size=64,
                shuffle=True,
                generator=g_dl,
                worker_init_fn=seed_worker,
                num_workers=0,
            )
        )

    global_model = MLP().to(DEVICE)

    rounds = 20
    local_lr = 0.05
    local_steps = 40

    # passos pequenos para medir dL "realista"
    alphas = [0.01, 0.10]

    print(f"DEVICE={DEVICE}")
    print("Rodada | Lval(w) | por cliente: cos(Δw,-g) | sat=tanh(||Δw||/c) | prox=cos*sat | score=(-g)·Δw | dL@0.01 | dL@0.10")
    print("        (cos/prox > 0 tende a ser 'a favor'; score>0 idem; dL<0 = melhorou val do servidor)")

    for t in range(1, rounds + 1):
        # loss do servidor antes
        L0 = eval_loss(global_model, val_loader, max_batches=10)

        # gradiente de referência no servidor (em w)
        gref = server_reference_grad(global_model, val_loader, batches=10)  # ∇L
        desc = (-gref).detach()  # direção de descida
        desc_norm = desc / (desc.norm() + 1e-12)

        w_flat = flatten_params(global_model).clone()

        # calcula Δw de todos os clientes primeiro (pra pegar a mediana das normas)
        deltas = []
        norms = []
        for loader in client_loaders:
            dw = local_train_delta(global_model, loader, lr=local_lr, steps=local_steps)
            deltas.append(dw)
            norms.append(dw.norm().item())

        norms_t = torch.tensor(norms, device=DEVICE)
        c = float(norms_t.median().item() + 1e-12)  # escala robusta (mediana)

        parts = []
        for i, dw in enumerate(deltas):
            # 1) direção
            cosv = cosine(dw, desc_norm)

            # 2) força saturada
            sat = float(torch.tanh(dw.norm() / c).item())

            # 3) score final da sua fórmula
            prox = cosv * sat

            # 4) "tangente"/força útil (1ª ordem)
            score = float(torch.dot(desc, dw).item())  # >0 tende a ajudar

            # 5) dL realista com passo pequeno (α)
            dLs = []
            for a in alphas:
                tmp = copy.deepcopy(global_model).to(DEVICE)
                load_flat_params_(tmp, w_flat + (a * dw))
                La = eval_loss(tmp, val_loader, max_batches=10)
                dLs.append(La - L0)  # negativo = melhorou

            parts.append(
                f"c{i}: cos={cosv:+.3f}, sat={sat:.3f}, prox={prox:+.3f}, "
                f"score={score:+.2e}, dL@{alphas[0]:.2f}={dLs[0]:+.4f}, dL@{alphas[1]:.2f}={dLs[1]:+.4f}"
            )

        print(f"{t:>6d} | {L0:>7.4f} | " + " | ".join(parts))

        # FedAvg no global (como antes)
        avg_dw = torch.stack(deltas, dim=0).mean(dim=0)
        load_flat_params_(global_model, w_flat + avg_dw)

    print("\nDone.")

if __name__ == "__main__":
    main()