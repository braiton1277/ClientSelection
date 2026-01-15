## seleçao de clientes com a logica da metrica (5 clientes)

import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================
# Reprodutibilidade TOTAL
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
# Modelo simples (MLP)
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
# Utils: flatten params/grads
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
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm() + eps)).item()


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
def eval_acc(model: nn.Module, loader: DataLoader, max_batches: int = 50) -> float:
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


def server_reference_grad(model: nn.Module, val_loader: DataLoader, batches: int = 10) -> torch.Tensor:
    """
    g_ref = ∇ L_val(w) (aponta para AUMENTAR a loss).
    Direção de descida: -g_ref.
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


# ============================
# Split não-IID em 5 clientes por rótulo
# ============================
def make_5_clients_noniid_indices(train_ds) -> List[List[int]]:
    groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

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
# Política baseada na métrica
# ============================
def select_by_metric(
    prox: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
    k: int,
    tau: float = 0.5,
    eps: float = 0.10,
    lam_stale: float = 0.05,
    gamma_streak: float = 0.5,
) -> List[int]:
    """
    prox: score por cliente (pode ser negativo)
    staleness: rodadas desde última seleção
    streak: quantas seleções seguidas

    Retorna k índices amostrados sem reposição.
    """
    n = len(prox)

    # 1) só o que é "a favor" (resto pode voltar via eps + staleness)
    u = np.maximum(0.0, prox).astype(np.float64)

    # 2) bônus por staleness (cobertura)
    u = u * (1.0 + lam_stale * staleness)

    # 3) penalidade por repetir demais seguidas
    u = u * (gamma_streak ** streak)

    # evita tudo zerar
    u = u + 1e-6

    # 4) softmax com temperatura (mais especializado = tau menor)
    logits = u / max(1e-12, tau)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()

    # 5) mistura com uniforme (exploração)
    p = (1.0 - eps) * p + eps * (1.0 / n)

    # amostra sem reposição
    p_t = torch.tensor(p, dtype=torch.float32)
    sel = torch.multinomial(p_t, num_samples=k, replacement=False).tolist()
    return sel


def update_staleness_streak(staleness: np.ndarray, streak: np.ndarray, selected: List[int]):
    n = len(staleness)
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    # staleness: quem não foi escolhido aumenta
    staleness[~sel_mask] += 1
    staleness[sel_mask] = 0

    # streak: quem foi escolhido incrementa, resto zera
    streak[~sel_mask] = 0
    streak[sel_mask] += 1


# ============================
# Rodada FL: calcula deltas, scores, seleciona, aplica FedAvg
# ============================
def compute_deltas_and_prox(
    model: nn.Module,
    client_loaders: List[DataLoader],
    val_loader: DataLoader,
    local_lr: float,
    local_steps: int,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    """
    Retorna:
      deltas: lista de Δw_i
      prox: score prox_i = cos(Δw, -gref_norm) * tanh(||Δw||/c)
    """
    # grad de referência no servidor
    gref = server_reference_grad(model, val_loader, batches=10)
    desc = (-gref).detach()
    desc_norm = desc / (desc.norm() + 1e-12)

    # deltas de todos clientes
    deltas = []
    norms = []
    for loader in client_loaders:
        dw = local_train_delta(model, loader, lr=local_lr, steps=local_steps)
        deltas.append(dw)
        norms.append(dw.norm().item())

    c = float(np.median(norms) + 1e-12)

    prox = []
    for dw in deltas:
        cosv = cosine(dw, desc_norm)
        sat = float(torch.tanh(dw.norm() / c).item())
        prox.append(cosv * sat)

    return deltas, np.array(prox, dtype=np.float32)


def apply_fedavg(model: nn.Module, deltas: List[torch.Tensor], selected: List[int]):
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


# ============================
# Experimento: Random vs Métrica
# ============================
def run_experiment(
    rounds: int = 50,
    k_select: int = 3,
    local_lr: float = 0.05,
    local_steps: int = 40,
    print_every: int = 5,
):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    # servidor: val balanceado + test (maior)
    val_idxs = make_server_val_balanced(test_ds, per_class=200)  # 2000
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

    # 5 clientes não-IID
    client_idxs = make_5_clients_noniid_indices(train_ds)
    client_loaders = []
    for idxs in client_idxs:
        idxs = idxs[:4000]
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

    # modelos: mesmo init
    base = MLP().to(DEVICE)
    model_rand = copy.deepcopy(base).to(DEVICE)
    model_met  = copy.deepcopy(base).to(DEVICE)

    n = len(client_loaders)
    staleness_r = np.zeros(n, dtype=np.float32)
    streak_r    = np.zeros(n, dtype=np.int32)
    staleness_m = np.zeros(n, dtype=np.float32)
    streak_m    = np.zeros(n, dtype=np.int32)

    hist = {
        "rand_loss": [],
        "rand_acc":  [],
        "met_loss":  [],
        "met_acc":   [],
    }

    print(f"DEVICE={DEVICE} | rounds={rounds} | K={k_select} | local_steps={local_steps} | local_lr={local_lr}")
    print("Comparando: RANDOM vs MÉTRICA(prox + staleness + streak + eps)")

    for t in range(1, rounds + 1):
        # avaliação (antes da rodada)
        l_r = eval_loss(model_rand, val_loader, max_batches=20)
        a_r = eval_acc(model_rand, test_loader, max_batches=80)
        l_m = eval_loss(model_met,  val_loader, max_batches=20)
        a_m = eval_acc(model_met,  test_loader, max_batches=80)

        hist["rand_loss"].append(l_r)
        hist["rand_acc"].append(a_r)
        hist["met_loss"].append(l_m)
        hist["met_acc"].append(a_m)

        # ---- RANDOM ----
        deltas_r, prox_r = compute_deltas_and_prox(model_rand, client_loaders, val_loader, local_lr, local_steps)
        selected_r = random.sample(range(n), k_select)
        apply_fedavg(model_rand, deltas_r, selected_r)
        update_staleness_streak(staleness_r, streak_r, selected_r)

        # ---- MÉTRICA ----
        deltas_m, prox_m = compute_deltas_and_prox(model_met, client_loaders, val_loader, local_lr, local_steps)
        selected_m = select_by_metric(
            prox=prox_m,
            staleness=staleness_m,
            streak=streak_m,
            k=k_select,
            tau=0.5,          # ↓ => mais especializado
            eps=0.10,         # exploração mínima
            lam_stale=0.05,   # cobertura
            gamma_streak=0.5  # não repetir demais
        )
        apply_fedavg(model_met, deltas_m, selected_m)
        update_staleness_streak(staleness_m, streak_m, selected_m)

        if t % print_every == 0:
            print(
                f"[round {t:3d}] "
                f"RAND: loss_val={l_r:.4f} acc_test={a_r*100:.2f}% sel={selected_r} | "
                f"MET:  loss_val={l_m:.4f} acc_test={a_m*100:.2f}% sel={selected_m} prox={np.round(prox_m,3)}"
            )

    # finais
    print("\nFinais:")
    print(f"RAND final: loss_val={hist['rand_loss'][-1]:.4f} acc_test={hist['rand_acc'][-1]*100:.2f}%")
    print(f"MET  final: loss_val={hist['met_loss'][-1]:.4f} acc_test={hist['met_acc'][-1]*100:.2f}%")

    return hist


if __name__ == "__main__":
    run_experiment(
        rounds=50,
        k_select=3,
        local_lr=0.05,
        local_steps=40,
        print_every=5
    )