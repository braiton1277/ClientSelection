#No contexto da seleção de clientes em aprendizado federado, a dinâmica do ambiente tende a 
# ser não estacionária, pois a relação entre um mesmo estado observado e a melhor decisão pode mudar a
# o longo das rodadas. Isso ocorre principalmente porque o modelo global evolui a cada 
# agregação, alterando a contribuição esperada de cada cliente (um cliente que ajuda muito no início pode
#  se tornar redundante depois, ou vice-versa), além de fatores como variações de disponibilidade, 
# latência e custo de comunicação e possíveis mudanças na distribuição dos dados locais (drift) ao
#  longo do tempo. Como consequência, as distribuições de recompensa e de transição associadas a (𝑠,𝑎)
#  podem variar com t, e, se o estado não incorporar informações sobre o “momento” do treinamento
#  (por exemplo, métricas do modelo global ou o índice da rodada), o problema pode ainda parecer
#  mais instável por estado parcialmente observado, exigindo estratégias que lidem com essa não 
# estacionariedade para manter o aprendizado robusto.



from dataclasses import dataclass

import numpy as np


@dataclass
class EnvConfig:
    n_clients: int = 20
    n_features: int = 6
    k_select: int = 5                 # quantos clientes por episódio
    nonstationary_every: int = 2000   # muda "mundo" (w) a cada X steps globais
    drift_std: float = 0.15           # intensidade do drift de w
    noise_std: float = 0.10           # ruído ao gerar clientes
    reward_scale: float = 1.0
    penalty_repeat: float = 0.5       # penalidade se escolher cliente repetido
    seed: int = 0


class ClientSelectionEnv:
    """
    Estado (observação):
      - features dos clientes (N x F) achatado
      - selected_mask (N)
      - global_level (1)
      => obs dim = N*F + N + 1

    Ação:
      - inteiro 0..N-1 (escolher cliente)

    Recompensa:
      - depende do score do cliente (dot(features, w))
      - penaliza escolher cliente repetido
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # nomes só pra debug
        base_names = ["quality", "diversity", "data", "latency", "compute", "noise"]
        self.feat_names = base_names[: cfg.n_features]

        self.w = self._init_w(cfg.n_features)

        self.t = 0              # step global
        self.step_in_ep = 0     # step do episódio
        self.global_level = 0.0

        self.clients = None
        self.selected_mask = None

    def _init_w(self, F):
        # qualidade/diversidade/dados/compute positivos, latency/noise negativos
        w = np.zeros(F, dtype=np.float32)
        for i, name in enumerate(self.feat_names):
            w[i] = -1.0 if name in ("latency", "noise") else 1.0
        w = w / (np.linalg.norm(w) + 1e-8)
        return w.astype(np.float32)

    def _maybe_drift(self):
        # drift periódico para tornar não-estacionário
        if self.cfg.nonstationary_every > 0 and (self.t % self.cfg.nonstationary_every == 0) and self.t > 0:
            drift = self.rng.normal(0.0, self.cfg.drift_std, size=self.cfg.n_features).astype(np.float32)
            self.w = (self.w + drift).astype(np.float32)
            self.w = self.w / (np.linalg.norm(self.w) + 1e-8)

    def _sample_clients(self):
        # clientes ~ N(0,1) + ruído, depois sigmoid -> (0,1)
        X = self.rng.normal(0.0, 1.0, size=(self.cfg.n_clients, self.cfg.n_features)).astype(np.float32)
        X += self.rng.normal(0.0, self.cfg.noise_std, size=X.shape).astype(np.float32)
        X = 1.0 / (1.0 + np.exp(-X))
        return X.astype(np.float32)

    def reset(self):
        self.step_in_ep = 0
        self.selected_mask = np.zeros(self.cfg.n_clients, dtype=np.float32)
        self.clients = self._sample_clients()
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.clients.reshape(-1),
                self.selected_mask,
                np.array([self.global_level], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def step(self, action: int):
        """
        Retorna: obs, reward, done, info
        """
        self.t += 1
        self.step_in_ep += 1
        self._maybe_drift()

        info = {}

        if action < 0 or action >= self.cfg.n_clients:
            reward = -1.0
            done = True
            info["reason"] = "invalid_action"
            return self._get_obs(), float(reward), done, info

        if self.selected_mask[action] == 1.0:
            reward = -self.cfg.penalty_repeat
            done = False
            info["reason"] = "repeat"
            return self._get_obs(), float(reward), done, info

        self.selected_mask[action] = 1.0

        # score do cliente no mundo atual
        score = float(self.clients[action] @ self.w)

        # recompensa suave do score (mapeia score -> [-1, +1])
        gain = 1.0 / (1.0 + np.exp(-3.0 * score))        # 0..1
        reward = self.cfg.reward_scale * (gain - 0.5) * 2.0

        # só pra debug (pode tirar depois)
        self.global_level = float(np.clip(self.global_level + 0.05 * reward, -5.0, 5.0))

        done = (self.selected_mask.sum() >= self.cfg.k_select)
        info["score"] = score
        info["gain"] = float(gain)
        return self._get_obs(), float(reward), done, info

    @property
    def obs_dim(self):
        return self.cfg.n_clients * self.cfg.n_features + self.cfg.n_clients + 1

    @property
    def n_actions(self):
        return self.cfg.n_clients


# -------------------------
# Prints / Debug
# -------------------------
def print_clients(env: ClientSelectionEnv, topk=10, show_all=False, title="CLIENTS"):
    clients = env.clients
    N, F = clients.shape
    mask = env.selected_mask.astype(int)

    scores = clients @ env.w
    order = np.argsort(-scores)
    idxs = order if show_all else order[: min(topk, N)]

    print(f"\n[{title}] step_global={env.t}  step_ep={env.step_in_ep}  global={env.global_level:.3f}")
    print("w (mundo):", np.round(env.w, 3))
    header = "idx sel  score   " + "  ".join([f"{n:>10}" for n in env.feat_names])
    print(header)
    print("-" * len(header))
    for i in idxs:
        row = clients[i]
        print(
            f"{i:>3}  {mask[i]:>1}  {scores[i]:>6.3f}  "
            + "  ".join([f"{v:>10.3f}" for v in row])
        )
    picked = np.where(mask == 1)[0]
    print("Selecionados:", picked.tolist() if len(picked) else "nenhum")


# -------------------------
# Demo rápido: reset + alguns steps
# -------------------------
if __name__ == "__main__":
    cfg = EnvConfig(
        n_clients=12,
        n_features=6,
        k_select=4,
        nonstationary_every=10,   # pequeno só pra você VER o drift rápido
        drift_std=0.25,
        noise_std=0.10,
        seed=0,
    )

    env = ClientSelectionEnv(cfg)
    obs = env.reset()
    print(f"[info] obs_dim={env.obs_dim} | n_actions={env.n_actions}")
    print_clients(env, topk=12, show_all=True, title="RESET")

    # Escolhe sempre o melhor cliente pelo score (só pra debug)
    for step in range(6):
        scores = env.clients @ env.w
        # pega o melhor ainda não selecionado
        candidates = np.where(env.selected_mask == 0)[0]
        best = candidates[np.argmax(scores[candidates])]

        obs, r, done, info = env.step(int(best))
        print(f"\n>> step {step+1}: action={best} | reward={r:.3f} | info={info}")
        print_clients(env, topk=12, show_all=True, title="AFTER STEP")

        if done:
            print("\n[done] episódio terminou (selecionou K clientes).")
            break