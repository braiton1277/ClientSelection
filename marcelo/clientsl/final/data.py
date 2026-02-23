from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset, Subset


# ============================
# Switchable deterministic TARGETED label flipping
# ============================
class SwitchableTargetedLabelFlipSubset(Dataset):
    """
    Pré-computa por amostra:
        u[i] ~ U(0,1)
        flipped_label[i] via um mapeamento fixo (targeted)
    Flip em tempo de execução:
        se enabled and u[i] < attack_rate
    Determinístico por amostra; attack_rate controla a fração atacada.
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


# ============================
# Server balanced validation (from TRAIN)
# ============================
def make_server_val_balanced(
    ds, per_class: int = 200, n_classes: int = 10, seed: int = 0
) -> List[int]:
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
def make_clients_dirichlet_indices(
    train_ds,
    n_clients: int = 50,
    alpha: float = 0.3,
    seed: int = 123,
    n_classes: int = 10,
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
                clients[cid].extend(idxs[start: start + c])
                start += c

    for cid in range(n_clients):
        rng.shuffle(clients[cid])

    return clients
