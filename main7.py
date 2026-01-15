#doubledqn + PER chegou em 96 e 95%

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

l1 = 64
l2 = 150
l3 = 100
l4 = 4

# -------------------------
# Função de teste
# -------------------------
def test_model(model, mode="random", display=True, max_moves=50):
    i = 0
    test_game = Gridworld(mode=mode)

    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state = torch.from_numpy(state_).float()

    if display:
        print("Initial State:")
        print(test_game.display())

    status = 1  # 1=in progress, 2=won, 0=lost
    model.eval()
    while status == 1:
        with torch.no_grad():
            qval = model(state)  # [1,4]
            action_ = torch.argmax(qval, dim=1).item()
        action = action_set[action_]

        if display:
            print(f"Move #: {i}; Taking action: {action}")

        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()

        if display:
            print(test_game.display())

        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print(f"Game won! Reward: {reward}")
            else:
                status = 0
                if display:
                    print(f"Game LOST. Reward: {reward}")

        i += 1
        if i > max_moves:
            if display:
                print("Game lost; too many moves.")
            break

    return True if status == 2 else False


# -------------------------
# PER Buffer (proportional)
# -------------------------
class PERBuffer:
    """
    PER simples:
      P(i) ∝ priority_i^alpha
      w_i = (N * P(i))^-beta, normalizado por max(w)
    """
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
# Main
# -------------------------
if __name__ == "__main__":
    # ----- Model (online) -----
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
    )

    # ----- Target network -----
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    # ----- Training setup -----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Huber loss por item (pra usar com PER weights)
    huber = nn.SmoothL1Loss(reduction="none")

    gamma = 0.95
    epsilon = 1.0
    eps_end = 0.1

    epochs = 5000
    losses = []

    # ----- PER hyperparams -----
    mem_size = 8000
    batch_size = 128
    per_alpha = 0.6
    beta_start = 0.4
    beta_end = 1.0
    per_eps = 1e-3

    replay = PERBuffer(capacity=mem_size, alpha=per_alpha, eps=per_eps)

    max_moves = 50
    sync_freq = 50
    j = 0  # contador global de steps (pra beta)

    # -------------------------
    # TREINO
    # -------------------------
    for ep in range(epochs):
        print(ep)
        epsilon = max(eps_end, epsilon - (1.0 - eps_end) / epochs)

        game = Gridworld(size=4, mode="random")

        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()

        mov = 0
        done = False

        while not done:
            j += 1
            mov += 1

            # ação (ε-greedy)
            qval = model(state1)  # [1,4]
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = torch.argmax(qval, dim=1).item()

            action = action_set[action_]
            game.makeMove(action)

            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()

            reward = float(game.reward())
            done = (reward != -1) or (mov >= max_moves)

            # salva no replay (sem grafo)
            replay.add((state1.detach(), action_, reward, state2.detach(), bool(done)))
            state1 = state2

            # treino
            if len(replay) >= batch_size:
                total_steps = epochs * max_moves
                frac = min(1.0, j / total_steps)
                beta = beta_start + frac * (beta_end - beta_start)

                minibatch, idxs, weights_np = replay.sample(batch_size, beta=beta)
                s1_list, a_list, r_list, s2_list, d_list = zip(*minibatch)

                state1_batch = torch.cat(s1_list, dim=0)  # [B,64]
                state2_batch = torch.cat(s2_list, dim=0)  # [B,64]

                action_batch = torch.as_tensor(a_list, dtype=torch.long)        # [B]
                reward_batch = torch.as_tensor(r_list, dtype=torch.float32)     # [B]
                done_batch   = torch.as_tensor(d_list, dtype=torch.float32)     # [B]
                weights      = torch.from_numpy(weights_np).float()             # [B]

                Q1 = model(state1_batch)  # [B,4]
                X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B]

                # ----- Double DQN target -----
                with torch.no_grad():
                    Q2_online = model(state2_batch)                 # [B,4]
                    best_actions = Q2_online.argmax(dim=1)          # [B]

                    Q2_target = model2(state2_batch)                # [B,4]
                    chosen_Q = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # [B]

                    Y = reward_batch + gamma * (1.0 - done_batch) * chosen_Q

                # ----- PER loss (weighted) com HUBER -----
                loss_per_item = huber(X, Y)                # [B]
                loss = (weights * loss_per_item).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # update priorities
                td_error = (Y - X)
                new_prios = td_error.detach().abs().cpu().numpy() + per_eps
                replay.update_priorities(idxs, new_prios)

            # sync target
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())
                model2.eval()

        if (ep + 1) % 100 == 0 and len(losses) > 0:
            print(f"epoch {ep+1}/{epochs} | last loss {losses[-1]:.4f} | eps {epsilon:.3f}")

    losses = np.array(losses)

    # -------------------------
    # TESTE: win rate
    # -------------------------
    max_games = 1000
    wins = 0
    for _ in range(max_games):
        if test_model(model, mode="random", display=False, max_moves=max_moves):
            wins += 1

    print(f"Games played: {max_games}, # of wins: {wins}")
    print(f"Win percentage: {wins/max_games:.3f}")

    