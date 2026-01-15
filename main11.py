import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Reprodutibilidade
# ----------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------
# Cliente simulado
# ----------------------------
@dataclass
class Client:
    cid: int
    quality: float   # "qualidade" real (oculta)
    noniid: float    # 0..1
    data_size: int
    cost: float


# ----------------------------
# Ambiente FL simulado
# ----------------------------
class FederatedSimEnv:
    """
    Ambiente não-estacionário leve:
    - drift no baseline e pequenas perturbações nas qualidades
    - reward = improvement - lam_cost * total_cost
    """
    def __init__(
        self,
        n_clients: int = 100,
        k: int = 10,
        nonstationary: bool = True,
        drift_every: int = 200,
        drift_scale: float = 0.08,
        device: str = "cpu",
    ):
        self.n_clients = n_clients
        self.k = k
        self.nonstationary = nonstationary
        self.drift_every = drift_every
        self.drift_scale = drift_scale
        self.device = device

        self.t = 0
        self.clients: List[Client] = self._init_clients()
        self.global_quality_bias = 0.0

        self.last_gain = np.zeros(n_clients, dtype=np.float32)
        self.last_cost = np.zeros(n_clients, dtype=np.float32)
        self.staleness = np.zeros(n_clients, dtype=np.float32)

    def _init_clients(self) -> List[Client]:
        clients = []
        for i in range(self.n_clients):
            quality = float(np.random.normal(loc=0.0, scale=1.0))
            noniid = float(np.random.rand())
            data_size = int(np.clip(np.random.lognormal(mean=5.5, sigma=0.6), 50, 20000))
            cost = float(np.clip(0.2 + 0.00005 * data_size + np.random.rand() * 0.5, 0.2, 2.5))
            clients.append(Client(i, quality, noniid, data_size, cost))
        return clients

    def _maybe_drift(self):
        if not self.nonstationary:
            return
        if self.t > 0 and (self.t % self.drift_every == 0):
            self.global_quality_bias += float(np.random.normal(0.0, self.drift_scale))
            for c in self.clients:
                c.quality += float(np.random.normal(0.0, self.drift_scale * 0.3))

    def get_client_features(self) -> np.ndarray:
        feats = np.zeros((self.n_clients, 6), dtype=np.float32)
        for i, c in enumerate(self.clients):
            feats[i, 0] = np.log1p(c.data_size)
            feats[i, 1] = c.cost
            feats[i, 2] = c.noniid
            feats[i, 3] = self.last_gain[i]
            feats[i, 4] = self.last_cost[i]
            feats[i, 5] = self.staleness[i]
        return feats

    def step(self, selected: List[int], lam_cost: float = 0.15) -> Tuple[float, dict]:
        assert len(selected) == self.k, f"esperado K={self.k}, veio {len(selected)}"

        self._maybe_drift()

        chosen_mask = np.zeros(self.n_clients, dtype=bool)
        chosen_mask[selected] = True
        self.staleness[~chosen_mask] += 1.0
        self.staleness[chosen_mask] = 0.0

        qualities = np.array([self.clients[i].quality for i in selected], dtype=np.float32)
        noniids = np.array([self.clients[i].noniid for i in selected], dtype=np.float32)
        costs = np.array([self.clients[i].cost for i in selected], dtype=np.float32)
        data_sizes = np.array([self.clients[i].data_size for i in selected], dtype=np.float32)

        base_gain = float(qualities.mean() + self.global_quality_bias)
        noniid_penalty = float(0.6 * noniids.mean())
        size_bonus = float(0.10 * (np.sqrt(data_sizes).mean() / 100.0))
        noise = float(np.random.normal(0.0, 0.25))

        improvement = base_gain + size_bonus - noniid_penalty + noise
        total_cost = float(costs.sum())

        reward = improvement - lam_cost * total_cost

        for idx in selected:
            self.last_gain[idx] = 0.7 * self.last_gain[idx] + 0.3 * improvement
            self.last_cost[idx] = 0.7 * self.last_cost[idx] + 0.3 * self.clients[idx].cost

        self.t += 1

        info = {
            "improvement": improvement,
            "total_cost": total_cost,
            "avg_quality": float(qualities.mean()),
            "avg_noniid": float(noniids.mean()),
            "avg_log_size": float(np.log1p(data_sizes).mean()),
            "bias": self.global_quality_bias,
        }
        return reward, info


# ----------------------------
# Rede do bandit (logit por cliente)
# ----------------------------
class BanditNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # logit escalar
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [N]


# ----------------------------
# Replay buffer (janela curta e amostragem enviesada ao recente)
# ----------------------------
class Replay:
    """
    Guarda tuplas: (feats, selected_idx_list, logp, entropy, reward)
    Mantém janela curta (não-estacionário) e amostra com viés pro recente.
    """
    def __init__(self, capacity: int = 5000, recent_bias: float = 0.7):
        self.buf = deque(maxlen=capacity)
        self.capacity = capacity
        self.recent_bias = recent_bias  # prob de pegar do terço mais recente

    def __len__(self):
        return len(self.buf)

    def add(self, item):
        self.buf.append(item)

    def sample(self, batch_size: int):
        n = len(self.buf)
        if n == 0:
            return []
        bs = min(batch_size, n)

        # mistura: com prob recent_bias, pega do terço mais recente
        recent_start = int(n * 2 / 3)
        idxs = []
        for _ in range(bs):
            if n >= 3 and random.random() < self.recent_bias:
                idxs.append(random.randrange(recent_start, n))
            else:
                idxs.append(random.randrange(0, n))

        return [self.buf[i] for i in idxs]


# ----------------------------
# Seleção K: Categorical sequencial sem reposição (com temperatura)
# ----------------------------
def select_k_clients_logprob(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Seleciona K sem reposição amostrando sequencialmente de uma Categorical.
    Retorna:
      selected (list[int])
      logp_total (tensor escalar)
      entropy_avg (tensor escalar)  (entropia média das escolhas)
    """
    assert logits.ndim == 1
    n = logits.numel()
    k = min(k, n)

    selected = []
    logp_total = torch.zeros((), device=logits.device)
    entropy_acc = torch.zeros((), device=logits.device)

    masked_logits = logits.clone()

    for _ in range(k):
        scaled = masked_logits / max(1e-8, float(temperature))
        dist = torch.distributions.Categorical(logits=scaled)
        a = dist.sample()  # índice
        selected.append(int(a.item()))
        logp_total = logp_total + dist.log_prob(a)
        entropy_acc = entropy_acc + dist.entropy()

        # impede repetir: mascara com -inf
        masked_logits[a] = -1e9

    entropy_avg = entropy_acc / k
    return selected, logp_total, entropy_avg


# ----------------------------
# Treino do bandit: REINFORCE com log-prob + adv normalizado + entropia + replay curto
# ----------------------------
def train_bandit(
    rounds: int = 3000,
    n_clients: int = 100,
    k: int = 10,
    lr: float = 1e-3,
    lam_cost: float = 0.15,
    print_every: int = 100,
    eval_every: int = 200,
    eval_games: int = 200,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = FederatedSimEnv(
        n_clients=n_clients,
        k=k,
        nonstationary=True,
        drift_every=250,
        drift_scale=0.10,
        device=device,
    )

    feats0 = env.get_client_features()
    in_dim = feats0.shape[1]
    model = BanditNet(in_dim=in_dim, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # baseline EMA (média e variância) pra normalizar adv
    beta = 0.99
    baseline = 0.0
    var = 1.0

    # entropia (exploração)
    ent_coef = 1e-2

    # temperatura e schedule (menos temperatura = mais greed)
    temp_start, temp_end = 1.5, 0.6
    temp_decay_rounds = max(1, rounds // 2)

    # replay curto
    replay = Replay(capacity=5000, recent_bias=0.75)
    batch_size = 32
    updates_per_round = 1  # 1-3 geralmente ok

    reward_hist, imp_hist = [], []

    def temperature_at(t: int) -> float:
        if t >= temp_decay_rounds:
            return temp_end
        frac = t / temp_decay_rounds
        return temp_start + (temp_end - temp_start) * frac

    # ----------------------------
    # Loop principal
    # ----------------------------
    for t in range(rounds):
        feats = env.get_client_features()
        x = torch.from_numpy(feats).float().to(device)

        logits = model(x)  # [N]
        temp = temperature_at(t)

        # amostra K e pega log-prob/entropia
        selected, logp, ent = select_k_clients_logprob(logits, k=k, temperature=temp)

        # executa ambiente
        reward, info = env.step(selected, lam_cost=lam_cost)

        # baseline EMA (mean/var) e vantagem normalizada + clip
        baseline = beta * baseline + (1.0 - beta) * reward
        delta = reward - baseline
        var = beta * var + (1.0 - beta) * (delta ** 2)
        std = float(np.sqrt(var + 1e-8))
        adv = float(delta / std)
        adv = float(np.clip(adv, -5.0, 5.0))

        # guarda transição no replay
        replay.add((feats, selected, float(logp.detach().cpu().item()), float(ent.detach().cpu().item()), reward))

        # ----------------------------
        # updates (com replay)
        # ----------------------------
        for _ in range(updates_per_round):
            batch = replay.sample(batch_size)
            if len(batch) == 0:
                break

            loss_total = torch.zeros((), device=device)

            for feats_b, sel_b, logp_b, ent_b, reward_b in batch:
                # recomputa logits/logp/entropia com parâmetros atuais (on-policy aproximado)
                xb = torch.from_numpy(feats_b).float().to(device)
                logits_b = model(xb)

                # reconstrói logp e entropia do conjunto escolhido via seleção sequencial "forçada"
                # (aproximação: computa log-prob do conjunto em ordem fixa sel_b)
                masked = logits_b.clone()
                logp_re = torch.zeros((), device=device)
                ent_re = torch.zeros((), device=device)

                for idx in sel_b:
                    dist = torch.distributions.Categorical(logits=masked / max(1e-8, float(temp)))
                    a = torch.as_tensor(idx, device=device)
                    logp_re = logp_re + dist.log_prob(a)
                    ent_re = ent_re + dist.entropy()
                    masked[a] = -1e9
                ent_re = ent_re / max(1, len(sel_b))

                # vantagem (usa baseline atual — funciona ok com janela curta)
                delta_b = reward_b - baseline
                adv_b = float(delta_b / std)
                adv_b = float(np.clip(adv_b, -5.0, 5.0))
                adv_t = torch.as_tensor(adv_b, dtype=torch.float32, device=device)

                # REINFORCE + entropia
                loss = -(adv_t * logp_re) - ent_coef * ent_re
                loss_total = loss_total + loss

            loss_total = loss_total / len(batch)

            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

        reward_hist.append(reward)
        imp_hist.append(info["improvement"])

        # prints
        if (t + 1) % print_every == 0:
            avg_r = float(np.mean(reward_hist[-print_every:]))
            avg_imp = float(np.mean(imp_hist[-print_every:]))
            print(
                f"[round {t+1:5d}/{rounds}] "
                f"temp={temp:.2f} | reward(avg)={avg_r:+.3f} | imp(avg)={avg_imp:+.3f} | "
                f"bias={info['bias']:+.3f} | cost={info['total_cost']:.2f} | noniid={info['avg_noniid']:.2f}"
            )

        # avaliação: learned (temp baixa) vs random
        if (t + 1) % eval_every == 0:
            def run_policy(policy: str) -> float:
                rewards = []
                for _ in range(eval_games):
                    feats_eval = env.get_client_features()
                    x_eval = torch.from_numpy(feats_eval).float().to(device)
                    with torch.no_grad():
                        logits_eval = model(x_eval)

                    if policy == "learned":
                        sel_eval, _, _ = select_k_clients_logprob(logits_eval, k=k, temperature=0.2)
                    else:
                        sel_eval = random.sample(range(n_clients), k)

                    r_eval, _ = env.step(sel_eval, lam_cost=lam_cost)
                    rewards.append(r_eval)
                return float(np.mean(rewards))

            learned_r = run_policy("learned")
            random_r = run_policy("random")

            print(f"  [eval @ round {t+1}] learned_avg_reward={learned_r:+.3f} | random_avg_reward={random_r:+.3f}")

            feats_eval = env.get_client_features()
            x_eval = torch.from_numpy(feats_eval).float().to(device)
            with torch.no_grad():
                sc = model(x_eval).detach().cpu().numpy()
            top = np.argsort(sc)[::-1][:10].tolist()
            print(f"  [eval] top-10 clients by logit: {', '.join(map(str, top))}")

    return model, env


# ----------------------------
# Rodar
# ----------------------------
if __name__ == "__main__":
    model, env = train_bandit(
        rounds=3000,
        n_clients=100,
        k=10,
        lr=1e-3,
        lam_cost=0.15,
        print_every=100,
        eval_every=200,
        eval_games=200,
        device=None,
    )

    print("\nDone.")