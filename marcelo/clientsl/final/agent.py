import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE


# ============================
# PER buffer (joint transitions)
# ============================
class PrioritizedReplayJoint:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        d_in: int,
        alpha: float = 0.6,
        eps: float = 1e-3,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.obs  = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.act  = np.zeros((capacity, n_agents), dtype=np.uint8)
        self.r    = np.zeros((capacity,), dtype=np.float32)
        self.obs2 = np.zeros((capacity, n_agents, d_in), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.p = np.zeros((capacity,), dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.max_p = 1.0

        self.rng = np.random.default_rng(int(seed))

    def add(self, obs, act, r, obs2, done: bool):
        self.obs[self.ptr] = obs.astype(np.float32)
        self.act[self.ptr] = act.astype(np.uint8)
        self.r[self.ptr] = float(r)
        self.obs2[self.ptr] = obs2.astype(np.float32)
        self.done[self.ptr] = 1.0 if done else 0.0

        self.p[self.ptr] = self.max_p
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        bs = min(int(batch_size), self.n)
        assert bs > 0

        pri = self.p[: self.n].astype(np.float64)
        probs = (pri + self.eps) ** self.alpha
        s = probs.sum()
        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)

        idx = self.rng.choice(self.n, size=bs, replace=False, p=probs)
        w = (self.n * probs[idx]) ** (-beta)
        w = w / (w.max() + 1e-12)

        return (
            self.obs[idx],
            self.act[idx],
            self.r[idx],
            self.obs2[idx],
            self.done[idx],
            idx.astype(np.int64),
            w.astype(np.float32),
        )

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray):
        td_abs = np.asarray(td_abs, dtype=np.float32)
        self.p[idx] = td_abs + self.eps
        self.max_p = float(max(self.max_p, float(td_abs.max(initial=0.0))))


# ============================
# Q-network (MLP por agente)
# ============================
class AgentMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================
# VDN selector (top-K + Double-DQN)
# ============================
class VDNSelector:
    def __init__(
        self,
        n_agents: int,
        d_in: int,
        k_select: int,
        hidden: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gamma: float = 0.90,
        grad_clip: float = 1.0,
        target_sync_every: int = 20,
        buf_size: int = 20000,
        batch_size: int = 128,
        train_steps: int = 20,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 4000,
        per_eps: float = 1e-3,
        double_dqn: bool = True,
        seed: int = 0,
    ):
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.k_select = int(k_select)
        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(target_sync_every)
        self.double_dqn = bool(double_dqn)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_steps = int(per_beta_steps)

        self.q = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt = AgentMLP(d_in=d_in, hidden=hidden).to(DEVICE)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)

        self.buf = PrioritizedReplayJoint(
            capacity=buf_size,
            n_agents=self.n_agents,
            d_in=self.d_in,
            alpha=per_alpha,
            eps=per_eps,
            seed=int(seed) + 12345,
        )

        self._train_calls = 0
        self.total_updates = 0
        self.total_samples_drawn = 0

        self.py_rng = random.Random(int(seed) + 777)
        self.np_rng = np.random.default_rng(int(seed) + 999)

    def _beta(self) -> float:
        t = min(self._train_calls, self.per_beta_steps)
        frac = t / max(1, self.per_beta_steps)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    @torch.no_grad()
    def _q_all_agents(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        return self.q(x).detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        self.q.eval()
        x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        return self.q(x).detach().cpu().numpy().astype(np.float32)

    def select_topk_actions(
        self,
        obs: np.ndarray,
        eps: float = 0.15,
        swap_m: int = 2,
        force_random: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        n = obs.shape[0]
        K = min(self.k_select, n)
        q = self._q_all_agents(obs)

        if force_random:
            sel = self.py_rng.sample(range(n), K)
            a = np.zeros(n, dtype=np.uint8)
            a[sel] = 1
            return a, sel

        adv = q[:, 1] - q[:, 0]
        sel = np.argsort(adv)[::-1][:K].tolist()

        if swap_m > 0 and (self.np_rng.random() < eps):
            swap_m = min(swap_m, K)
            sel_set = set(sel)
            not_sel = [i for i in range(n) if i not in sel_set]
            if len(not_sel) > 0:
                out = self.py_rng.sample(sel, swap_m)
                inn = self.py_rng.sample(not_sel, min(swap_m, len(not_sel)))
                sel2 = sel.copy()
                for o, i_new in zip(out, inn):
                    sel2[sel2.index(o)] = i_new
                sel2 = list(dict.fromkeys(sel2))
                while len(sel2) < K:
                    cand = self.py_rng.randrange(n)
                    if cand not in sel2:
                        sel2.append(cand)
                sel = sel2[:K]

        a = np.zeros(n, dtype=np.uint8)
        a[sel] = 1
        return a, sel

    def add_transition(self, obs, act, r, obs2, done: bool):
        self.buf.add(obs=obs, act=act, r=r, obs2=obs2, done=done)

    def train(
        self, batch_size: Optional[int] = None, train_steps: Optional[int] = None
    ) -> Optional[float]:
        bs_req = int(batch_size) if batch_size is not None else self.batch_size
        steps = int(train_steps) if train_steps is not None else self.train_steps

        if self.buf.n < max(32, bs_req):
            return None

        beta = self._beta()
        self.q.train()
        losses = []

        for _ in range(steps):
            ob, ac, rw, ob2, dn, idx, w_is = self.buf.sample(batch_size=bs_req, beta=beta)
            B = ob.shape[0]
            self.total_samples_drawn += B

            obs  = torch.tensor(ob,  dtype=torch.float32, device=DEVICE)
            act  = torch.tensor(ac,  dtype=torch.long,    device=DEVICE)
            r    = torch.tensor(rw,  dtype=torch.float32, device=DEVICE)
            obs2 = torch.tensor(ob2, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(dn,  dtype=torch.float32, device=DEVICE)
            w    = torch.tensor(w_is, dtype=torch.float32, device=DEVICE)

            N = self.n_agents
            q_cur = self.q(obs.reshape(B * N, self.d_in)).reshape(B, N, 2)
            q_a = q_cur.gather(2, act.unsqueeze(2)).squeeze(2)
            q_tot = q_a.sum(dim=1)

            with torch.no_grad():
                q2_online = self.q(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)
                q2_tgt    = self.q_tgt(obs2.reshape(B * N, self.d_in)).reshape(B, N, 2)

                K = min(self.k_select, N)
                adv2 = (
                    (q2_online[:, :, 1] - q2_online[:, :, 0])
                    if self.double_dqn
                    else (q2_tgt[:, :, 1] - q2_tgt[:, :, 0])
                )
                top_idx = adv2.topk(K, dim=1).indices
                a2 = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
                a2.scatter_(1, top_idx, 1)

                q2_a = q2_tgt.gather(2, a2.unsqueeze(2)).squeeze(2)
                q2_tot = q2_a.sum(dim=1)
                y = r + (1.0 - done) * self.gamma * q2_tot

            td_abs = (q_tot - y).detach().abs().cpu().numpy().astype(np.float32)
            loss = (w * F.smooth_l1_loss(q_tot, y, reduction="none")).mean()

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.opt.step()

            self.buf.update_priorities(idx, td_abs)
            losses.append(float(loss.item()))
            self.total_updates += 1

        self._train_calls += 1
        if self._train_calls % self.target_sync_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(np.mean(losses)) if losses else None


# ============================
# Estado VDN (d=5)
# [bias, proj_mom, probe_loss_now, staleness_norm, streak_norm]
# ============================
def build_context_matrix_vdn(
    projection_mom: np.ndarray,
    probe_now: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
) -> np.ndarray:
    proj  = projection_mom.astype(np.float32)
    probe = probe_now.astype(np.float32)

    s = staleness.astype(np.float32)
    sn = (s / (s.max() + 1e-6)).astype(np.float32)

    t = streak.astype(np.float32)
    tn = np.clip(t / 5.0, 0.0, 1.0).astype(np.float32)

    bias = np.ones(proj.shape[0], dtype=np.float32)
    return np.stack([bias, proj, probe, sn, tn], axis=1).astype(np.float32)
