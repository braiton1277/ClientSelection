# DoubleDQN + PER + NoisyNet (5x5) fez 65% talvez pq recopensa eh muito esparsa usar moisy n adianta mt
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Gridworld import Gridworld

# -------------------------
# Util
# -------------------------
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def get_state(game, l1, device="cpu", noise=0.0):
    """Lê o estado do env e retorna tensor [1, l1]."""
    s = game.board.render_np().reshape(1, -1).astype(np.float32)
    if s.shape[1] != l1:
        raise ValueError(f"Estado mudou de dimensão! esperado {l1}, veio {s.shape[1]}")
    if noise > 0.0:
        s = s + np.random.rand(1, l1).astype(np.float32) * noise
    return torch.from_numpy(s).float().to(device)


# -------------------------
# Função de teste (win/lose)
# -------------------------
def test_model(model, grid_size=5, mode="random", max_moves=200, device="cpu", l1=None):
    test_game = Gridworld(size=grid_size, mode=mode)

    if l1 is None:
        l1 = test_game.board.render_np().reshape(-1).size

    state = get_state(test_game, l1, device=device, noise=0.0)

    status = 1
    mov = 0
    model.eval()

    while status == 1:
        with torch.no_grad():
            qval = model(state)               # [1,4]
            action_ = torch.argmax(qval, 1).item()

        test_game.makeMove(action_set[action_])
        state = get_state(test_game, l1, device=device, noise=0.0)

        reward = test_game.reward()
        mov += 1

        if reward != -1:
            status = 2 if reward > 0 else 0
        if mov >= max_moves:
            status = 0

    return status == 2


# -------------------------
# PER Buffer (proportional)
# -------------------------
class PERBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(self, transition):
        self.data[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.data[i] for i in idxs]

        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return samples, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, new_prios):
        new_prios = np.asarray(new_prios, dtype=np.float32)
        new_prios = np.maximum(new_prios, self.eps)
        self.priorities[idxs] = new_prios
        self.max_priority = max(self.max_priority, float(new_prios.max()))


# -------------------------
# NoisyLinear (factorized gaussian)
# -------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # parâmetros determinísticos (mu)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # parâmetros do ruído (sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # buffers para epsilons (amostrados a cada forward/reset)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # init recomendado no paper: mu ~ U[-1/sqrt(in), 1/sqrt(in)]
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # sigma constante / sqrt(in)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        eps_in = self._f(eps_in)
        eps_out = self._f(eps_out)

        self.weight_epsilon.copy_(eps_out.ger(eps_in))  # outer product
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# -------------------------
# NoisyDQN (MLP com NoisyLinear)
# -------------------------
class NoisyDQN(nn.Module):
    def __init__(self, in_dim, hidden1=256, hidden2=256, n_actions=4, sigma_init=0.5):
        super().__init__()
        self.fc1 = NoisyLinear(in_dim, hidden1, sigma_init=sigma_init)
        self.fc2 = NoisyLinear(hidden1, hidden2, sigma_init=sigma_init)
        self.fc3 = NoisyLinear(hidden2, n_actions, sigma_init=sigma_init)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GRID_SIZE = 5

    # Descobre a dimensão real do estado
    tmp_game = Gridworld(size=GRID_SIZE, mode="random")
    l1 = tmp_game.board.render_np().reshape(-1).size
    print(f"[info] GRID_SIZE={GRID_SIZE} | state_dim (l1) = {l1}")

    n_actions = 4

    # Model (online) - NoisyNet
    model = NoisyDQN(in_dim=l1, hidden1=256, hidden2=256, n_actions=n_actions, sigma_init=0.5).to(device)

    # Target network
    model2 = copy.deepcopy(model).to(device)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    huber = nn.SmoothL1Loss(reduction="none")

    gamma = 0.99

    epochs = 5000

    # PER hyperparams
    mem_size = 20000
    batch_size = 64
    per_alpha = 0.6
    beta_start = 0.4
    beta_end = 1.0
    per_eps = 1e-3
    replay = PERBuffer(capacity=mem_size, alpha=per_alpha, eps=per_eps)

    max_moves = 200
    sync_freq = 200
    j = 0

    # -------------------------
    # TREINO
    # -------------------------
    for ep in range(epochs):
        game = Gridworld(size=GRID_SIZE, mode="random")
        state1 = get_state(game, l1, device=device, noise=0.0)

        mov = 0
        done = False

        while not done:
            j += 1
            mov += 1

            # (Noisy) reamostra ruído para explorar
            model.train()
            model.reset_noise()

            with torch.no_grad():
                qval = model(state1)
                action_ = torch.argmax(qval, dim=1).item()

            game.makeMove(action_set[action_])

            state2 = get_state(game, l1, device=device, noise=0.0)

            reward = float(game.reward())
            done = (reward != -1) or (mov >= max_moves)

            replay.add((state1.detach(), action_, reward, state2.detach(), float(done)))
            state1 = state2

            # treino
            if len(replay) >= batch_size:
                total_steps = epochs * max_moves
                frac = min(1.0, j / total_steps)
                beta = beta_start + frac * (beta_end - beta_start)

                minibatch, idxs, weights_np = replay.sample(batch_size, beta=beta)
                s1_list, a_list, r_list, s2_list, d_list = zip(*minibatch)

                s1 = torch.cat(s1_list, dim=0).to(device)
                s2 = torch.cat(s2_list, dim=0).to(device)
                a  = torch.as_tensor(a_list, dtype=torch.long, device=device)
                r  = torch.as_tensor(r_list, dtype=torch.float32, device=device)
                d  = torch.as_tensor(d_list, dtype=torch.float32, device=device)
                w  = torch.from_numpy(weights_np).float().to(device)

                model.train()
                model.reset_noise()
                Q1 = model(s1)
                X = Q1.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN target (reamostra ruído no online também)
                    model.reset_noise()
                    Q2_online = model(s2)
                    best_actions = Q2_online.argmax(dim=1)

                    Q2_target = model2(s2)
                    chosen_Q = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

                    Y = r + gamma * (1.0 - d) * chosen_Q

                td_error = (Y - X)
                loss_per_item = huber(X, Y)
                loss = (w * loss_per_item).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                new_prios = td_error.detach().abs().cpu().numpy() + per_eps
                replay.update_priorities(idxs, new_prios)

            # sync target
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())
                model2.eval()

        # winrate print a cada 200 episódios
        if (ep + 1) % 200 == 0:
            wins = 0
            games = 200
            for _ in range(games):
                if test_model(model, grid_size=GRID_SIZE, mode="random",
                              max_moves=max_moves, device=device, l1=l1):
                    wins += 1
            print(f"ep {ep+1}/{epochs} | winrate {wins/games:.3f} | replay {len(replay)}")

    # teste final
    wins = 0
    max_games = 1000
    for _ in range(max_games):
        if test_model(model, grid_size=GRID_SIZE, mode="random",
                      max_moves=max_moves, device=device, l1=l1):
            wins += 1
    print(f"Games played: {max_games}, # of wins: {wins}")
    print(f"Win percentage: {wins/max_games:.3f}")