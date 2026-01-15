#adicionado dueling, 2nstep atrapalha tudo  


import copy
import random

import numpy as np
import torch
import torch.nn as nn

from Gridworld import Gridworld

# -------------------------
# Util
# -------------------------
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
l1, l4 = 64, 4  # state dim, n_actions


# -------------------------
# Dueling DQN (sem NoisyNet)
# -------------------------
class DuelingDQN(nn.Module):
    def __init__(self, in_dim=64, hidden=128, n_actions=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)                         # [B,1]
        a = self.adv(z)                           # [B,A]
        q = v + (a - a.mean(dim=1, keepdim=True))  # dueling combine
        return q


# -------------------------
# PER Buffer (proportional)
# -------------------------
class PERBuffer:
    def __init__(self, capacity, alpha=0.5, eps=1e-3):
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
# Teste
# -------------------------
def test_model(model, mode="random", max_moves=50, device="cpu"):
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state = torch.from_numpy(state_).float().to(device)

    status = 1
    mov = 0
    model.eval()
    while status == 1:
        with torch.no_grad():
            qval = model(state)
            action_ = torch.argmax(qval, dim=1).item()
        action = action_set[action_]
        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state = torch.from_numpy(state_).float().to(device)

        reward = test_game.reward()
        mov += 1
        if reward != -1:
            status = 2 if reward > 0 else 0
        if mov >= max_moves:
            status = 0

    return status == 2


# -------------------------
# Treino (Double + PER + Dueling)  <-- SEM N-STEP
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DuelingDQN(in_dim=64, hidden=128, n_actions=4).to(device)
    model2 = copy.deepcopy(model).to(device)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    huber = nn.SmoothL1Loss(reduction="none")

    gamma = 0.95

    # PER
    mem_size = 5000
    batch_size = 128
    per_alpha = 0.5
    per_eps = 1e-3
    beta_start, beta_end = 0.4, 1.0
    replay = PERBuffer(capacity=mem_size, alpha=per_alpha, eps=per_eps)

    epochs = 5000
    max_moves = 50

    # target sync
    sync_freq = 50
    j = 0

    # ε-greedy
    epsilon = 1.0
    eps_end = 0.1

    for ep in range(epochs):
        print(ep)
        epsilon = max(eps_end, epsilon - (1.0 - eps_end) / epochs)

        game = Gridworld(size=4, mode="random")

        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float().to(device)

        mov = 0
        done = False

        while not done:
            j += 1
            mov += 1

            # ação (ε-greedy)
            with torch.no_grad():
                qval = model(state1)
                if random.random() < epsilon:
                    action_ = random.randint(0, 3)
                else:
                    action_ = torch.argmax(qval, dim=1).item()

            action = action_set[action_]
            game.makeMove(action)

            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float().to(device)

            reward = float(game.reward())
            done = (reward != -1) or (mov >= max_moves)

            # salva no replay (1-step)
            replay.add((state1.detach(), action_, reward, state2.detach(), float(done)))
            state1 = state2

            # treino
            if len(replay) >= batch_size:
                total_steps = epochs * max_moves
                frac = min(1.0, j / total_steps)
                beta = beta_start + frac * (beta_end - beta_start)

                minibatch, idxs, weights_np = replay.sample(batch_size, beta=beta)
                s1_list, a_list, r_list, s2_list, d_list = zip(*minibatch)

                s1 = torch.cat(s1_list, dim=0).to(device)  # [B,64]
                s2 = torch.cat(s2_list, dim=0).to(device)  # [B,64]
                a  = torch.as_tensor(a_list, dtype=torch.long, device=device)     # [B]
                r  = torch.as_tensor(r_list, dtype=torch.float32, device=device) # [B]
                d  = torch.as_tensor(d_list, dtype=torch.float32, device=device) # [B]
                w  = torch.from_numpy(weights_np).float().to(device)              # [B]

                # Q(s,a)
                model.train()
                Q1 = model(s1)                              # [B,4]
                X = Q1.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    # Double DQN: online escolhe, target avalia
                    q_online = model(s2)                     # [B,4]
                    best_a = q_online.argmax(dim=1)          # [B]

                    q_tgt = model2(s2)                       # [B,4]
                    chosen_q = q_tgt.gather(1, best_a.unsqueeze(1)).squeeze(1)  # [B]

                    Y = r + gamma * (1.0 - d) * chosen_q

                td_error = (Y - X)
                loss_per_item = huber(X, Y)                 # [B]
                loss = (w * loss_per_item).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

                # update priorities
                new_prios = td_error.detach().abs().cpu().numpy() + per_eps
                replay.update_priorities(idxs, new_prios)

            # sync target
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())
                model2.eval()

        if (ep + 1) % 200 == 0:
            wins = 0
            games = 200
            for _ in range(games):
                if test_model(model, mode="random", max_moves=max_moves, device=device):
                    wins += 1
            print(f"ep {ep+1}/{epochs} | winrate {wins/games:.3f} | eps {epsilon:.3f} | replay {len(replay)}")

    # teste final
    wins = 0
    max_games = 1000
    for _ in range(max_games):
        if test_model(model, mode="random", max_moves=max_moves, device=device):
            wins += 1
    print(f"Games played: {max_games}, # of wins: {wins}")
    print(f"Win percentage: {wins/max_games:.3f}")