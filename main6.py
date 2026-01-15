import copy
import random
from collections import deque

import numpy as np
import torch
from matplotlib import pylab as plt

from Gridworld import Gridworld

# -------------------------
# Função de teste
# -------------------------

#dqn apenas com double dqn

action_set = {
 0: 'u',
 1: 'd',
 2: 'l',
 3: 'r',
}
l1 = 64
l2 = 150
l3 = 100
l4 = 4
def test_model(model, mode="random", display=True):
    i = 0
    test_game = Gridworld(mode=mode)

    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state = torch.from_numpy(state_).float()

    if display:
        print("Initial State:")
        print(test_game.display())

    status = 1  # 1=in progress, 2=won, 0=lost
    while status == 1:
        qval = model(state)  # [1,4]
        action_ = torch.argmax(qval, dim=1).item()
        action = action_set[action_]

        if display:
            print(f"Move #: {i}; Taking action: {action}")

        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
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
        #if i > 15:
        if i > max_moves:
            if display:
                print("Game lost; too many moves.")
            break

    return True if status == 2 else False


# -------------------------
# Execução principal
# -------------------------
if __name__ == "__main__":

    # IMPORTANTE: esses valores precisam existir no seu código:
    # l1,l2,l3,l4, gamma, epsilon, action_set
    # e action_set deve mapear 0..3 para ações do Gridworld.

    # ----- Model (online) -----
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )

    # ----- Target network -----
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    model2.eval()  # target só para inferência

    # ----- Training setup -----
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learning_rate = 1e-3

    gamma = 0.95
    
    epsilon = 1.0
    epochs = 5000
    losses = []

    mem_size = 1000
    batch_size = 128
    replay = deque(maxlen=mem_size)

    max_moves = 50
    sync_freq = 50
    j = 0

    # -------------------------
    # TREINO
    # -------------------------
    for i in range(epochs):
        print(i)
        epsilon = max(0.1, epsilon - 1/epochs)
        #epsilon = max(0.1, epsilon * 0.995)
        game = Gridworld(size=4, mode="random")

        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()

        status = 1
        mov = 0

        while status == 1:
            j += 1
            mov += 1

            qval = model(state1)  # [1,4]

            # epsilon-greedy
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = torch.argmax(qval, dim=1).item()

            action = action_set[action_]
            game.makeMove(action)

            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()

            reward = game.reward()
            #done = True if reward > 0 else False  # como no livro
            done = (reward != -1) or (mov >= max_moves)

            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            # update com minibatch (batch otimizado)
            if len(replay) >= batch_size:
                minibatch = random.sample(replay, batch_size)
                s1_list, a_list, r_list, s2_list, d_list = zip(*minibatch)

                state1_batch = torch.cat(s1_list, dim=0)  # [B,64]
                state2_batch = torch.cat(s2_list, dim=0)  # [B,64]

                action_batch = torch.as_tensor(a_list, dtype=torch.long)        # [B]
                reward_batch = torch.as_tensor(r_list, dtype=torch.float32)     # [B]
                done_batch   = torch.as_tensor(d_list, dtype=torch.float32)     # [B]

                Q1 = model(state1_batch)  # [B,4]
                #with torch.no_grad():
                    #Q2 = model2(state2_batch)          # [B,4]
                    #maxQ2 = Q2.max(dim=1).values       # [B]
                
                with torch.no_grad():
                    # 1) online escolhe a ação (argmax)
                    Q2_online = model(state2_batch)                 # [B,4]
                    best_actions = Q2_online.argmax(dim=1)          # [B]

                    # 2) target avalia essa ação
                    Q2_target = model2(state2_batch)                # [B,4]
                    chosen_Q = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # [B]

                #Y = reward_batch + gamma * (1.0 - done_batch) * maxQ2          # [B]
                
                Y = reward_batch + gamma * (1.0 - done_batch) * chosen_Q
                X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)         # [B]

                loss = loss_fn(X, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            # sync target network
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())
                model2.eval()

            # termina episódio
            if reward != -1 or mov > max_moves:
                status = 0

        # log leve
        if done:
            status = 0
        if (i + 1) % 100 == 0 and len(losses) > 0:
            print(f"epoch {i+1}/{epochs} | last loss {losses[-1]:.4f}")

    losses = np.array(losses)

    # -------------------------
    # TESTE: win rate
    # -------------------------
    max_games = 1000
    wins = 0
    for _ in range(max_games):
        if test_model(model, mode="random", display=False):
            wins += 1

    print(f"Games played: {max_games}, # of wins: {wins}")
    print(f"Win percentage: {wins/max_games:.3f}")