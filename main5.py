#Qlearning

import random
from collections import deque

import numpy as np
import torch
from matplotlib import pylab as plt

from Gridworld import Gridworld

l1 = 64
l2 = 150
l3 = 100
l4 = 4
model = torch.nn.Sequential(
 torch.nn.Linear(l1, l2),
 torch.nn.ReLU(),
 torch.nn.Linear(l2, l3),
 torch.nn.ReLU(),
 torch.nn.Linear(l3,l4)
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = 0.9
epsilon = 1.0


action_set = {
 0: 'u',
 1: 'd',
 2: 'l',
 3: 'r',
}


# epochs = 1000
# losses = []

# for i in range(epochs):
#     print(i)
#     game = Gridworld(size=4, mode="static")

#     state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
#     state1 = torch.from_numpy(state_).float()

#     status = 1
#     while status == 1:
#         qval = model(state1)
#         qval_ = qval.data.numpy()

#         if random.random() < epsilon:
#             action_ = np.random.randint(0, 4)
#         else:
#             action_ = np.argmax(qval_)

#         action = action_set[action_]
#         game.makeMove(action)

#         state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
#         state2 = torch.from_numpy(state2_).float()

#         reward = game.reward()

#         with torch.no_grad():
#             newQ = model(state2.reshape(1, 64))
#             maxQ = torch.max(newQ)

#         if reward == -1:
#             Y = reward + (gamma * maxQ)
#         else:
#             Y = reward

#         Y = torch.Tensor([Y]).detach()
#         X = qval[0, action_].unsqueeze(0) 

#         loss = loss_fn(X, Y)
#         optimizer.zero_grad()
#         loss.backward()
#         losses.append(loss.item())
#         optimizer.step()

#         state1 = state2

#         if reward != -1:
#             status = 0

#     if epsilon > 0.1:
#         epsilon -= (1 / epochs)


#dqn classico

epochs = 5000
losses = []

mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)

max_moves = 50

epochs = 3000
losses = []

mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)

max_moves = 50

for i in range(epochs):
    print(i)
    game = Gridworld(size=4, mode="random")

    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state1 = torch.from_numpy(state1_).float()

    status = 1
    mov = 0

    while status == 1:
        mov += 1

        qval = model(state1)
        qval_ = qval.detach().numpy()

        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = int(np.argmax(qval_))

        action = action_set[action_]
        game.makeMove(action)

        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2 = torch.from_numpy(state2_).float()

        reward = game.reward()
        done = True if reward > 0 else False  # (mantive igual ao seu; ajuste se tiver terminal de derrota)

        exp = (state1, action_, reward, state2, done)
        replay.append(exp)

        state1 = state2

        # ----- OPTIMIZED BATCHING (CPU) -----
        if len(replay) >= batch_size:
            minibatch = random.sample(replay, batch_size)
            s1_list, a_list, r_list, s2_list, d_list = zip(*minibatch)

            # s1/s2 são tensores [1,64] -> concat vira [B,64]
            state1_batch = torch.cat(s1_list, dim=0)
            state2_batch = torch.cat(s2_list, dim=0)

            action_batch = torch.as_tensor(a_list, dtype=torch.long)         # [B]
            reward_batch = torch.as_tensor(r_list, dtype=torch.float32)      # [B]
            done_batch   = torch.as_tensor(d_list, dtype=torch.float32)      # [B]

            Q1 = model(state1_batch)  # [B,4]
            with torch.no_grad():
                Q2 = model(state2_batch)               # [B,4]
                maxQ2 = Q2.max(dim=1).values           # [B]

            Y = reward_batch + gamma * (1.0 - done_batch) * maxQ2            # [B]
            X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze(1)           # [B]

            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        # -----------------------------------

        if reward != -1 or mov > max_moves:
            status = 0

losses = np.array(losses)



def test_model(model, mode="random", display=True):
    i = 0
    test_game = Gridworld(mode=mode)

    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()

    if display:
        print("Initial State:")
        print(test_game.display())

    status = 1  # 1=in progress, 2=won, 0=lost
    while status == 1:
        qval = model(state)
        qval_ = qval.data.numpy()

        action_ = np.argmax(qval_)
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
        if i > 15:
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


# if __name__ == "__main__":
#     # (garanta que action_set, model, etc. já existem aqui)
#     win = test_model(model, mode="random", display=True)
#     print("WIN?" , win)


if __name__ == "__main__":

   
    max_games = 1000
    wins = 0
    for _ in range(max_games):
        if test_model(model, mode="random", display=False):
            wins += 1
    print(f"Games played: {max_games}, # of wins: {wins}")
    print(f"Win percentage: {wins/max_games:.3f}")