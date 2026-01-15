#contextual bandit
#estacionário, bandit_matrix fixa, a melhor açao em cada estado nao muda
# Estacionário quer dizer que, para cada par 
# (s,a), a distribuição da recompensa não muda com o tempo


#DETERMINISTICO X ESTOCÁSTICO
# Determinístico

# Dado um estado
# s e uma ação 𝑎
# o que acontece é sempre igual.
# R(s,a) é um número fixo.


# Ex.: “se eu escolher o braço 3 no estado 7, ganho sempre 5 pontos”.

# Estocástico (não determinístico)

# Mesmo fixando s e 𝑎, o resultado é sorteado de uma distribuição.

# Ex.: “se eu escolher o braço 3 no estado 7, tenho 70% de chance de ganhar, 30% de não ganhar”.

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

N, D_in, H, D_out = 1, 10, 100, 10


# class ContextBandit:
#     def __init__(self, arms=10, change_step=5000, seed=None):
#         self.arms = arms
#         self.change_step = change_step
#         self.t = 0
#         if seed is not None:
#             np.random.seed(seed)
#         self.init_distribution(arms)
#         self.update_state()
#         self.t = 0
    
#     def init_distribution(self, arms):
#         self.bandit_matrix = np.random.rand(arms,arms)

#     def reward(self, prob):
#         reward = 0
#         for i in range(self.arms):
#             if random.random() < prob:
#                 reward += 1
#         return reward
    

#     def get_state(self):
#         return self.state
    

#     def update_state(self):
#         self.state = np.random.randint(0,self.arms)


#     def get_reward(self, arm):
#         return self.reward(self.bandit_matrix[self.get_state()][arm])
    
#     def maybe_change(self):
#         if self.t == self.change_step:
#             self.init_distribution(self.arms)   # nova matriz inteira
    
#     def choose_arm(self, arm):
#         reward = self.get_reward(arm)
#         self.update_state()
#         self.t += 1
#         return reward


class ContextBandit:
    def __init__(self, arms=10, update_every=1000):
        self.arms = arms
        self.update_every = update_every
        self.t = 0  # contador de rodadas
        self.init_distribution(arms)
        self.update_state()
        #self.next_update = self.update_every
    
    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms, arms)

    def maybe_update_matrix(self):
        # atualiza a cada update_every rodadas (exceto na rodada 0)
        if self.t > 0 and (self.t % self.update_every == 0):
            self.init_distribution(self.arms)

        # if self.t > 0 and self.t == self.next_update:
        #     self.init_distribution(self.arms)
        #     self.next_update += random.randint(1, self.update_every)
            

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state
    
    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])
    
    def choose_arm(self, arm):
        self.maybe_update_matrix()      # <<< aqui
        r = self.get_reward(arm)
        self.update_state()
        self.t += 1                     # <<< incrementa rodada
        return r
    
#trocar relu por softplus talvez?
# retirar e deixar linear no final?

model = torch.nn.Sequential(
 torch.nn.Linear(D_in, H),
 torch.nn.ReLU(),
 torch.nn.Linear(H, D_out),
 #nn.Softplus(),
 
)
loss_fn = torch.nn.MSELoss()

def one_hot(N, pos, val=1):
 one_hot_vec = np.zeros(N)
 one_hot_vec[pos] = val
 return one_hot_vec

def softmax(av, tau=2.0):
    av = np.asarray(av, dtype=np.float64)
    z = av / tau
    z = z - np.max(z)         # estabilidade numérica
    e = np.exp(z)
    return e / np.sum(e)


def print_policy_per_state(model, arms=10, tau=2):
    model.eval()
    with torch.no_grad():
        for s in range(arms):
            x = torch.tensor(one_hot(arms, s), dtype=torch.float32)
            logits = model(x)                    # (arms,)
            probs = torch.softmax(logits / tau, dim=0).cpu().numpy()
            print(f"state {s}: {np.round(probs, 3)} | best={probs.argmax()}")

# def train(env, arms, epochs=6000, learning_rate=1e-2): 
#     cur_state = torch.Tensor(one_hot(arms,env.get_state()))
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     rewards = []
#     for i in range(epochs):
#         y_pred = model(cur_state)
#         av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
#         av_softmax /= av_softmax.sum()
#         choice = np.random.choice(arms, p=av_softmax)
#         cur_reward = env.choose_arm(choice) 
#         one_hot_reward = y_pred.data.numpy().copy()
#         one_hot_reward[choice] = cur_reward
#         reward = torch.Tensor(one_hot_reward)
#         rewards.append(cur_reward)
#         loss = loss_fn(y_pred, reward)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         cur_state = torch.Tensor(one_hot(arms,env.get_state())) 
#     return np.array(rewards)


#aparamentemente esse proximo eh o melhor para o ambiente q muda 1x

# def train(env, arms, epochs=6000, learning_rate=1e-2, buffer_size=1000, batch_size=32, tau=2.0):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     rewards = []

#     # Replay buffer: guarda (state, action, reward)
#     buffer = deque(maxlen=buffer_size)

#     for i in range(epochs):
#         # Estado atual
#         s = env.get_state()
#         cur_state = torch.tensor(one_hot(arms, s), dtype=torch.float32)

#         # Forward
#         y_pred = model(cur_state)

#         # Policy (softmax)
#         with torch.no_grad():
#             probs = torch.softmax(y_pred / tau, dim=0).cpu().numpy()

#         # Escolhe ação e coleta recompensa
#         choice = np.random.choice(arms, p=probs)
#         cur_reward = env.choose_arm(choice)

#         rewards.append(cur_reward)
#         buffer.append((s, choice, cur_reward))

#         # Só treina quando tiver batch suficiente
#         if len(buffer) < batch_size:
#             continue

#         # Amostra batch do buffer
#         batch = random.sample(buffer, batch_size)

#         optimizer.zero_grad()
#         loss_total = 0.0

#         for sb, ab, rb in batch:
#             xb = torch.tensor(one_hot(arms, sb), dtype=torch.float32)
#             qb = model(xb)

#             # Target: igual sua lógica (clona e só troca a ação escolhida)
#             target = qb.detach().clone()
#             target[ab] = rb

#             loss_total = loss_total + loss_fn(qb, target)

#         loss_total.backward()
#         optimizer.step()

#     return np.array(rewards)

# troca de matriz a cada 500 o proximo funciona melhor (janela das ultimas 500)

# def train(env, arms, epochs=6000, learning_rate=0.01, buffer_size=500, batch_size=32, tau=1.5, recent_window=200):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     rewards = []
#     buffer = deque(maxlen=buffer_size)

#     for i in range(epochs):
#         s = env.get_state()
#         cur_state = torch.tensor(one_hot(arms, s), dtype=torch.float32)

#         y_pred = model(cur_state)

#         with torch.no_grad():
#             probs = torch.softmax(y_pred / tau, dim=0).cpu().numpy()

#         choice = np.random.choice(arms, p=probs)
#         cur_reward = env.choose_arm(choice)

#         rewards.append(cur_reward)
#         buffer.append((s, choice, cur_reward))

#         if len(buffer) < batch_size:
#             continue

#         # -------- Opção 1: amostra só da janela recente --------
#         buf_list = list(buffer)
#         recent = buf_list[-recent_window:] if len(buf_list) > recent_window else buf_list
#         batch = random.sample(recent, batch_size)
#         # ------------------------------------------------------

#         optimizer.zero_grad()
#         loss_total = 0.0

#         for sb, ab, rb in batch:
#             xb = torch.tensor(one_hot(arms, sb), dtype=torch.float32)
#             qb = model(xb)

#             target = qb.detach().clone()
#             target[ab] = rb

#             loss_total = loss_total + loss_fn(qb, target)

#         loss_total.backward()
#         optimizer.step()

#         if i % 500 == 0:
#             print(f"i={i} mean_last100={np.mean(rewards[-100:]):.2f}")

#     return np.array(rewards)

#janela com e greedy parece melhor que o softmax para janela de 500

def train(env, arms, epochs=6000, learning_rate=0.01, buffer_size=500, batch_size=32, epsilon=0.1, recent_window=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    buffer = deque(maxlen=buffer_size)

    for i in range(epochs):
        s = env.get_state()
        cur_state = torch.tensor(one_hot(arms, s), dtype=torch.float32)

        # 1. Previsão para escolha (sem gradiente para performance)
        with torch.no_grad():
            y_pred = model(cur_state)

        # 2. Epsilon-Greedy
        if np.random.rand() < epsilon:
            choice = np.random.randint(arms)
        else:
            choice = torch.argmax(y_pred).item()

        cur_reward = env.choose_arm(choice)
        rewards.append(cur_reward)
        buffer.append((s, choice, cur_reward))

        if len(buffer) < batch_size:
            continue

        # 3. Amostragem da janela recente
        buf_list = list(buffer)
        recent = buf_list[-recent_window:] if len(buf_list) > recent_window else buf_list
        batch = random.sample(recent, batch_size)

        # --- VETORIZAÇÃO (Otimização de Velocidade) ---
        
        # Converte a lista de tuplas do batch em tensores de uma vez só
        states_b = torch.stack([torch.tensor(one_hot(arms, x[0]), dtype=torch.float32) for x in batch])
        actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        optimizer.zero_grad()
        
        # Previsão paralela para todas as 32 amostras
        q_preds = model(states_b) 

        # Criamos o alvo (target) baseado nos valores atuais
        targets = q_preds.detach().clone()
        
        # Atualização em massa: 
        # range(batch_size) seleciona todas as linhas, actions_b seleciona as colunas específicas
        targets[range(len(batch)), actions_b] = rewards_b

        # Perda calculada sobre o batch inteiro
        loss = loss_fn(q_preds, targets)
        loss.backward()
        optimizer.step()
        
        # ----------------------------------------------

        if i % 500 == 0:
            print(f"i={i} mean_last100={np.mean(rewards[-100:]):.2f}")

    return np.array(rewards)

# def train(env, arms, epochs=6000, learning_rate=0.01, buffer_size=500, batch_size=32, epsilon=0.1, recent_window=200):
#     # Inicialização do otimizador
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     rewards = []
    
#     # Reduzi o buffer para focar apenas em dados frescos
#     buffer = deque(maxlen=buffer_size)

#     for i in range(epochs):
#         s = env.get_state()
#         cur_state = torch.tensor(one_hot(arms, s), dtype=torch.float32)

#         # 1. Escolha da ação (Epsilon-Greedy)
#         model.eval()
#         with torch.no_grad():
#             q_values = model(cur_state)
        
#         if np.random.rand() < epsilon:
#             choice = np.random.randint(arms)
#         else:
#             choice = torch.argmax(q_values).item()

#         # 2. Interação com o ambiente
#         cur_reward = env.choose_arm(choice)
#         rewards.append(cur_reward)
#         buffer.append((s, choice, cur_reward))

#         # --- NOVIDADE: Reset do Otimizador ---
#         # Se o ambiente mudou (i > 0 e múltiplo de env.update_every)
#         if i > 0 and i % env.update_every == 0:
#             # Reiniciamos o Adam para apagar o momentum antigo
#             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#             # Opcional: Limpar o buffer para não treinar com o passado
#             #buffer.clear()
#             print(f">>> Ambiente mudou no step {i}: Otimizador e Buffer resetados!")

#         if len(buffer) < batch_size:
#             continue

#         # 3. Treino (Janela Recente)
#         model.train()
#         buf_list = list(buffer)
#         recent = buf_list[-recent_window:] if len(buf_list) > recent_window else buf_list
#         batch = random.sample(recent, batch_size)

#         # Otimização: Preparando o batch inteiro de uma vez (vetorizado)
#         states_b = torch.stack([torch.tensor(one_hot(arms, x[0]), dtype=torch.float32) for x in batch])
#         actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)
#         rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)

#         optimizer.zero_grad()
        
#         # Previsão para o batch
#         q_preds = model(states_b) # Saída: [batch_size, arms]
        
#         # Criando o Target
#         targets = q_preds.detach().clone()
#         # Atualiza apenas o valor da ação que foi tomada com a recompensa recebida
#         targets[range(len(batch)), actions_b] = rewards_b

#         loss = loss_fn(q_preds, targets)
#         loss.backward()
        
#         # Gradient Clipping para evitar explosão após a mudança
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()

#         if i % 500 == 0:
#             mean_r = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
#             print(f"i={i} | Recompensa Média (últimas 100): {mean_r:.2f}")

#     return np.array(rewards)

# Aqui usa o desconto, mas tem q usar a janela tb, pois se nao as amostras do buffer irao atrapalhar,
# como ele vai amostrar do propria janela o alpha deve ser bem pequeno para ele nao esquecer dados bons
# da janela boa. Ainda n consegui deixar bom com a mudança a cada 500 da matriz

# def train(
#     env,
#     arms,
#     epochs=6000,
#     learning_rate=1e-2,
#     tau=2.0,
#     alpha=0.05,          # forgetting real (EMA)
#     buffer_size=2000,
#     batch_size=32,
#     recent_window=500
# ):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     rewards = []
#     buffer = deque(maxlen=buffer_size)

#     # EMA por (state, action): alvo "suavizado" e com esquecimento
#     Q_ema = np.zeros((arms, arms), dtype=np.float32)

#     for i in range(epochs):
#         # -------- coleta 1 transição --------
#         s = env.get_state()
#         x = torch.tensor(one_hot(arms, s), dtype=torch.float32)

#         y_pred = model(x)

#         with torch.no_grad():
#             probs = torch.softmax(y_pred / tau, dim=0).cpu().numpy()

#         a = np.random.choice(arms, p=probs)
#         r = env.choose_arm(a)

#         rewards.append(r)
#         buffer.append((s, a, r))

#         # atualiza EMA do par (s,a) observado
#         Q_ema[s, a] = (1 - alpha) * Q_ema[s, a] + alpha * r

#         # -------- treino com batch --------
#         if len(buffer) < batch_size:
#             continue

#         buf_list = list(buffer)
#         recent = buf_list[-recent_window:] if len(buf_list) > recent_window else buf_list
#         batch = random.sample(recent, batch_size)

#         optimizer.zero_grad()
#         loss_total = 0.0

#         for sb, ab, rb in batch:
#             xb = torch.tensor(one_hot(arms, sb), dtype=torch.float32)
#             qb = model(xb)

#             target = qb.detach().clone()
#             # alvo para a ação escolhida vem da EMA (não do qb)
#             target[ab] = float(Q_ema[sb, ab])

#             loss_total = loss_total + loss_fn(qb, target)

#         loss_total.backward()
#         optimizer.step()

#         if i % 500 == 0:
#             print(f"i={i} mean_last100={np.mean(rewards[-100:]):.2f}")

#     return np.array(rewards)


############MUDANÇA DE AMBIENTE - ESTACIONÁRIO #############################


# import random
# from collections import deque

# import numpy as np
# import torch

# # -----------------------------
# # Config
# # -----------------------------
# ARMS = 10
# CONTEXT_DIM = 5   # "Idade, Interesse em Esportes, Tecnologia, etc."
# H = 64            # Neurônios na camada oculta


# # -----------------------------
# # Ambiente (ESTACIONÁRIO)
# # -----------------------------
# class RichContextBanditStationary:
#     def __init__(self, arms=ARMS, context_dim=CONTEXT_DIM):
#         self.arms = arms
#         self.context_dim = context_dim
#         self.t = 0
#         self.init_distribution()  # fixa o "mundo"

#     def init_distribution(self):
#         # Matriz oculta fixa (não muda!)
#         self.hidden_weights = np.random.randn(self.context_dim, self.arms)

#     def get_state(self):
#         return np.random.randn(self.context_dim).astype(np.float32)

#     def get_reward(self, state, arm):
#         score = np.dot(state, self.hidden_weights[:, arm])
#         prob = 1 / (1 + np.exp(-score))  # prob de clique
#         return 1 if np.random.random() < prob else 0

#     def step(self, arm, state):
#         # estacionário: nunca muda
#         r = self.get_reward(state, arm)
#         self.t += 1
#         return r


# # -----------------------------
# # Modelo (Rede Neural)
# # Saída = logits (NÃO aplicar sigmoid aqui!)
# # -----------------------------
# model = torch.nn.Sequential(
#     torch.nn.Linear(CONTEXT_DIM, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, ARMS)
# )

# bce_logits = torch.nn.BCEWithLogitsLoss()


# # -----------------------------
# # Treino (ESTACIONÁRIO)
# # - Replay buffer com janela recente (mas no estacionário pode ser grande)
# # - Epsilon decay (explora mais no início, menos depois)
# # - BCE nos logits do braço escolhido
# # -----------------------------
# def train_stationary(
#     env,
#     epochs=30000,
#     lr=1e-3,
#     batch_size=64,
#     recent_window=5000,
#     eps_start=0.5,
#     eps_end=0.05,
#     eps_decay_steps=15000,
#     warmup_steps=2000,
#     print_every=1000
# ):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     rewards = []
#     buffer = deque(maxlen=recent_window)

#     def epsilon_at(step):
#         if step <= 0:
#             return eps_start
#         if step >= eps_decay_steps:
#             return eps_end
#         frac = step / eps_decay_steps
#         return eps_start + frac * (eps_end - eps_start)

#     for i in range(epochs):
#         s = env.get_state()
#         state_t = torch.from_numpy(s)  # float32

#         # política
#         model.eval()
#         with torch.no_grad():
#             logits = model(state_t)

#         eps = epsilon_at(i)

#         # warmup: explora totalmente no início
#         if i < warmup_steps or np.random.rand() < eps:
#             a = np.random.randint(env.arms)
#         else:
#             a = int(torch.argmax(logits).item())

#         r = env.step(a, s)
#         rewards.append(r)
#         buffer.append((s, a, r))

#         if len(buffer) < batch_size:
#             continue

#         # batch train
#         model.train()
#         batch = random.sample(list(buffer), batch_size)

#         states_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)  # (B, D)
#         actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)              # (B,)
#         rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)           # (B,)

#         optimizer.zero_grad()
#         all_logits = model(states_b)  # (B, ARMS)
#         chosen_logits = all_logits.gather(1, actions_b.view(-1, 1)).squeeze(1)  # (B,)

#         loss = bce_logits(chosen_logits, rewards_b)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#         optimizer.step()

#         if i % print_every == 0 and i > 0:
#             print(f"Step {i:5d} | eps={eps:.2f} | Reward médio (últimos 1000): {np.mean(rewards[-1000:]):.3f}")

#     return np.array(rewards)


# # -----------------------------
# # Teste estilo "acerto/erro" (TOP-1)
# # -----------------------------
# def test_top1(model, env, num_tests=12):
#     print("\n--- TESTE DE POLÍTICA (PREVISÃO vs REALIDADE) | TOP-1 ---")
#     model.eval()
#     with torch.no_grad():
#         for i in range(num_tests):
#             s = env.get_state()

#             # previsão (logits)
#             logits = model(torch.from_numpy(s))
#             predicted_arm = int(torch.argmax(logits).item())

#             # melhor real (argmax do score linear oculto)
#             real_scores = [np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)]
#             true_best_arm = int(np.argmax(real_scores))

#             status = "✅ ACERTO" if predicted_arm == true_best_arm else "❌ ERRO"
#             print(
#                 f"Teste {i+1:2d}: Perfil {np.round(s, 2)} | "
#                 f"Escolha: {predicted_arm:2d} | Melhor Real: {true_best_arm:2d} | {status}"
#             )


# # -----------------------------
# # (Opcional) Teste TOP-K (mais estável)
# # -----------------------------
# def test_topk(model, env, num_tests=12, k=3):
#     print(f"\n--- TESTE DE POLÍTICA (PREVISÃO vs REALIDADE) | TOP-{k} ---")
#     model.eval()
#     with torch.no_grad():
#         for i in range(num_tests):
#             s = env.get_state()

#             logits = model(torch.from_numpy(s)).numpy()
#             topk_pred = list(np.argsort(logits)[-k:][::-1])

#             real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
#             true_best_arm = int(np.argmax(real_scores))

#             status = "✅ ACERTO" if true_best_arm in topk_pred else "❌ ERRO"
#             print(
#                 f"Teste {i+1:2d}: Perfil {np.round(s, 2)} | "
#                 f"Top-{k}: {topk_pred} | Melhor Real: {true_best_arm:2d} | {status}"
#             )


######## NOVO MODELO NAO ESTACIONARIO ################

import random
from collections import deque

import numpy as np
import torch

# -----------------------------
# Config
# -----------------------------
ARMS = 10
CONTEXT_DIM = 5
H = 64



class RichContextBanditNonStationary:
    def __init__(self, arms=ARMS, context_dim=CONTEXT_DIM, update_every=1000):
        self.arms = arms
        self.context_dim = context_dim
        self.update_every = update_every
        self.t = 0
        self.world_id = 0
        self.init_distribution()

    def init_distribution(self):
        self.hidden_weights = np.random.randn(self.context_dim, self.arms)
        self.world_id += 1

    def get_state(self):
        return np.random.randn(self.context_dim).astype(np.float32)

    def get_reward(self, state, arm):
        score = np.dot(state, self.hidden_weights[:, arm])
        prob = 1 / (1 + np.exp(-score))
        return 1 if np.random.random() < prob else 0

    def step(self, arm, state):
        # muda o mundo ANTES de gerar recompensa
        changed = (self.t > 0 and self.t % self.update_every == 0)
        if changed:
            self.init_distribution()

        r = self.get_reward(state, arm)
        self.t += 1
        return r, changed


# -----------------------------
# Modelo (logits por braço)
# -----------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(CONTEXT_DIM, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, ARMS)
)

bce_logits = torch.nn.BCEWithLogitsLoss()


# -----------------------------
# Treino (NÃO-ESTACIONÁRIO)
# Estratégia:
# - replay window curta (esquece rápido)
# - epsilon com piso (nunca vai a ~0)
# - "boost" de exploração após mudança do mundo
# - limpa buffer e reseta Adam quando o mundo muda
# -----------------------------
def train_nonstationary(
    env,
    epochs=30000,
    lr=1e-3,
    batch_size=64,
    recent_window=400,        # curto = esquece rápido
    eps_start=0.5,
    eps_end=0.15,             # piso alto (não pode ir a ~0 no não-estacionário)
    eps_decay_steps=15000,
    warmup_steps=2000,
    boost_eps=0.6,            # exploração extra após mudança
    boost_steps=250,          # por quantos passos após mudança
    print_every=1000
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rewards = []
    buffer = deque(maxlen=recent_window)

    steps_since_change = 10**9  # grande no começo

    def epsilon_base(step):
        if step >= eps_decay_steps:
            return eps_end
        frac = step / eps_decay_steps
        return eps_start + frac * (eps_end - eps_start)

    for i in range(epochs):
        s = env.get_state()
        state_t = torch.from_numpy(s)

        # logits para exploração/exploração
        model.eval()
        with torch.no_grad():
            logits = model(state_t)

        # epsilon do momento (com boost após mudança)
        eps = epsilon_base(i)
        if steps_since_change < boost_steps:
            eps = max(eps, boost_eps)

        # warmup: explora bastante no início
        if i < warmup_steps or np.random.rand() < eps:
            a = np.random.randint(env.arms)
        else:
            a = int(torch.argmax(logits).item())

        r, changed = env.step(a, s)
        rewards.append(r)

        if changed:
            steps_since_change = 0
            buffer.clear()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            steps_since_change += 1

        buffer.append((s, a, r))

        if len(buffer) < batch_size:
            continue

        # treino em batch
        model.train()
        batch = random.sample(list(buffer), batch_size)

        states_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)  # (B, D)
        actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)              # (B,)
        rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)           # (B,)

        optimizer.zero_grad()
        all_logits = model(states_b)  # (B, ARMS)
        chosen_logits = all_logits.gather(1, actions_b.view(-1, 1)).squeeze(1)

        loss = bce_logits(chosen_logits, rewards_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if i % print_every == 0 and i > 0:
            print(
                f"Step {i:5d} | eps={eps:.2f} | "
                f"Reward médio (últimos 1000): {np.mean(rewards[-1000:]):.3f} | "
                f"world_id={env.world_id}"
            )

    return np.array(rewards)


# -----------------------------
# Testes "acerta/erra" (TOP-1 e TOP-K)
# Obs: isso é severo no não-estacionário; rode logo após treinar,
# e lembre que o mundo pode ter mudado recentemente.
# -----------------------------
def test_top1(model, env, num_tests=12):
    print("\n--- TESTE TOP-1 (ACERTO/ERRO) ---")
    model.eval()
    with torch.no_grad():
        for i in range(num_tests):
            s = env.get_state()

            logits = model(torch.from_numpy(s))
            pred_arm = int(torch.argmax(logits).item())

            real_scores = [np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)]
            best_arm = int(np.argmax(real_scores))

            status = "✅ ACERTO" if pred_arm == best_arm else "❌ ERRO"
            print(
                f"Teste {i+1:2d}: Perfil {np.round(s, 2)} | "
                f"Escolha: {pred_arm:2d} | Melhor Real: {best_arm:2d} | {status}"
            )


def test_topk(model, env, num_tests=12, k=2):
    print(f"\n--- TESTE TOP-{k} (ACERTO/ERRO) ---")
    model.eval()
    with torch.no_grad():
        for i in range(num_tests):
            s = env.get_state()

            logits = model(torch.from_numpy(s)).numpy()
            topk_pred = list(np.argsort(logits)[-k:][::-1])

            real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
            best_arm = int(np.argmax(real_scores))

            status = "✅ ACERTO" if best_arm in topk_pred else "❌ ERRO"
            print(
                f"Teste {i+1:2d}: Perfil {np.round(s, 2)} | "
                f"Top-{k}: {topk_pred} | Melhor Real: {best_arm:2d} | {status}"
            )



import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
ARMS = 10
CONTEXT_DIM = 5
H = 64


# ============================================================
# Ambiente NÃO-ESTACIONÁRIO (mudanças ALEATÓRIAS) e SEM SINAL
# - O agente NÃO recebe "changed"
# ============================================================
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

ARMS = 10
CONTEXT_DIM = 5
H = 64


# ============================================================
# Ambiente NÃO-ESTACIONÁRIO com DRIFT SUAVE (sem aviso)
# ============================================================
class RichContextBanditSmoothDrift:
    def __init__(self, arms=ARMS, context_dim=CONTEXT_DIM, drift_sigma=0.01):
        self.arms = arms
        self.context_dim = context_dim
        self.drift_sigma = drift_sigma
        self.t = 0
        self.hidden_weights = np.random.randn(self.context_dim, self.arms)

    def get_state(self):
        return np.random.randn(self.context_dim).astype(np.float32)

    def _drift(self):
        # random walk pequeno: W <- W + sigma * N(0,1)
        self.hidden_weights += self.drift_sigma * np.random.randn(self.context_dim, self.arms)

    def step(self, arm, state):
        # drift suave a cada passo (pode mudar para a cada N passos)
        self._drift()

        score = float(np.dot(state, self.hidden_weights[:, arm]))
        prob = 1 / (1 + np.exp(-score))
        r = 1 if np.random.random() < prob else 0

        self.t += 1
        return r


# ============================================================
# Detector Page-Hinkley (usando loss_step)
# ============================================================
class PageHinkley:
    def __init__(self, delta=0.001, threshold=6.0, alpha=0.99):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.t = 0

    def update(self, x):
        x = float(x)
        self.t += 1

        if self.t == 1:
            self.mean = x
        else:
            self.mean = self.alpha * self.mean + (1 - self.alpha) * x

        self.cum += (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)

        return (self.cum - self.min_cum) > self.threshold


# ============================================================
# Modelo
# ============================================================
model = torch.nn.Sequential(
    torch.nn.Linear(CONTEXT_DIM, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, ARMS)
)


# ============================================================
# Treino para DRIFT SUAVE
# Estratégia:
# - NÃO reseta agressivo
# - mantém eps piso + boost leve quando detector "acende"
# - só reseta se detector ficar disparando demais (mundo mudou muito)
# ============================================================
def train_smooth_drift(
    env,
    epochs=40000,
    lr=1e-3,
    batch_size=64,
    recent_window=1500,

    eps_base=0.2,        # piso permanente
    boost_eps=0.5,       # boost quando detector alerta
    boost_steps=300,

    ph_delta=0.001,
    ph_threshold=6.0,
    ph_alpha=0.99,

    # reset só se houver muitas detecções em pouco tempo
    reset_if_detections_in_last=8,
    reset_window=2000,

    print_every=1000
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = deque(maxlen=recent_window)
    rewards = []
    loss_hist = []

    detector = PageHinkley(delta=ph_delta, threshold=ph_threshold, alpha=ph_alpha)

    boost_left = 0
    detections = 0
    det_steps = deque(maxlen=reset_if_detections_in_last)  # guarda quando detectou

    for i in range(epochs):
        s = env.get_state()
        state_t = torch.from_numpy(s)

        model.eval()
        with torch.no_grad():
            logits = model(state_t)

        eps = boost_eps if boost_left > 0 else eps_base

        # epsilon-greedy (sempre tem exploração mínima)
        if np.random.rand() < eps:
            a = np.random.randint(env.arms)
        else:
            a = int(torch.argmax(logits).item())

        r = env.step(a, s)
        rewards.append(r)
        buffer.append((s, a, r))

        # surpresa do modelo
        with torch.no_grad():
            loss_step = F.binary_cross_entropy_with_logits(
                logits[a], torch.tensor(float(r))
            ).item()
        loss_hist.append(loss_step)

        # detector
        if detector.update(loss_step):
            detections += 1
            boost_left = boost_steps
            det_steps.append(i)
            detector.reset()  # evita cascata

        if boost_left > 0:
            boost_left -= 1

        # reset "forte" só se estiver detectando demais (mudança brusca)
        if len(det_steps) == det_steps.maxlen and (det_steps[-1] - det_steps[0]) < reset_window:
            buffer.clear()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            det_steps.clear()

        if len(buffer) < batch_size:
            continue

        model.train()
        batch = random.sample(list(buffer), batch_size)
        states_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        optimizer.zero_grad()
        all_logits = model(states_b)
        chosen_logits = all_logits.gather(1, actions_b.view(-1, 1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(chosen_logits, rewards_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if i % print_every == 0 and i > 0:
            print(
                f"Step {i:5d} | eps={'BOOST' if boost_left>0 else 'BASE '}={eps:.2f} | "
                f"Reward (últimos 1000): {np.mean(rewards[-1000:]):.3f} | "
                f"Loss_step (últimos 1000): {np.mean(loss_hist[-1000:]):.3f} | "
                f"detecções={detections}"
            )

    return np.array(rewards)


# -----------------------------
# Testes top-1 e top-2
# -----------------------------
def test_top1(model, env, num_tests=20):
    print("\n--- TESTE TOP-1 ---")
    model.eval()
    with torch.no_grad():
        hits = 0
        for i in range(num_tests):
            s = env.get_state()
            logits = model(torch.from_numpy(s))
            pred = int(torch.argmax(logits).item())

            real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
            best = int(np.argmax(real_scores))

            ok = (pred == best)
            hits += int(ok)
            print(f"Teste {i+1:2d}: escolha={pred} melhor_real={best} {'✅' if ok else '❌'}")
        print(f"Acertos: {hits}/{num_tests}")


def test_topk(model, env, num_tests=20, k=2):
    print(f"\n--- TESTE TOP-{k} ---")
    model.eval()
    with torch.no_grad():
        hits = 0
        for i in range(num_tests):
            s = env.get_state()
            logits = model(torch.from_numpy(s)).numpy()
            topk = [int(x) for x in np.argsort(logits)[-k:][::-1]]

            real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
            best = int(np.argmax(real_scores))

            ok = best in topk
            hits += int(ok)
            print(f"Teste {i+1:2d}: top{k}={topk} melhor_real={best} {'✅' if ok else '❌'}")
        print(f"Acertos: {hits}/{num_tests}")





# Treino para DRIFT SUAVE

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    env = RichContextBanditSmoothDrift(
        arms=ARMS,
        context_dim=CONTEXT_DIM,
        drift_sigma=0.01  # se quiser mais difícil, aumenta pra 0.03
    )

    print("Treino (drift suave, sem aviso)...")
    train_smooth_drift(
        env,
        epochs=40000,
        lr=1e-3,
        batch_size=64,
        recent_window=1500,
        eps_base=0.2,
        boost_eps=0.5,
        boost_steps=300,
        ph_delta=0.001,
        ph_threshold=6.0,
        ph_alpha=0.99,
        print_every=1000
    )

    test_top1(model, env, num_tests=20)
    test_topk(model, env, num_tests=20, k=2)


import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
ARMS = 10
CONTEXT_DIM = 5
H = 64


# ============================================================
# Ambiente NÃO-ESTACIONÁRIO "MEIO-TERMO"
# - fica estável por um tempo
# - em tempos aleatórios, faz uma mudança PARCIAL (não hard reset total)
#
# Ideia:
#   W <- (1 - mix) * W + mix * W_new + noise
# onde mix ~ 0.2..0.6 (mudança moderada)
# ============================================================
class RichContextBanditModerateChange:
    def __init__(
        self,
        arms=ARMS,
        context_dim=CONTEXT_DIM,
        min_change=600,
        max_change=1800,
        mix_low=0.25,
        mix_high=0.55,
        noise_sigma=0.02
    ):
        self.arms = arms
        self.context_dim = context_dim

        self.min_change = min_change
        self.max_change = max_change

        self.mix_low = mix_low
        self.mix_high = mix_high
        self.noise_sigma = noise_sigma

        self.t = 0
        self.hidden_weights = np.random.randn(self.context_dim, self.arms)
        self._schedule_next_change()

    def _schedule_next_change(self):
        self.next_change_at = self.t + np.random.randint(self.min_change, self.max_change + 1)

    def _moderate_change(self):
        W_new = np.random.randn(self.context_dim, self.arms)
        mix = np.random.uniform(self.mix_low, self.mix_high)  # quanto do "mundo novo" entra
        noise = self.noise_sigma * np.random.randn(self.context_dim, self.arms)

        self.hidden_weights = (1 - mix) * self.hidden_weights + mix * W_new + noise
        self._schedule_next_change()

    def get_state(self):
        return np.random.randn(self.context_dim).astype(np.float32)

    def step(self, arm, state):
        # muda de forma oculta, sem avisar
        if self.t > 0 and self.t >= self.next_change_at:
            self._moderate_change()

        score = float(np.dot(state, self.hidden_weights[:, arm]))
        prob = 1 / (1 + np.exp(-score))
        r = 1 if np.random.random() < prob else 0

        self.t += 1
        return r


# ============================================================
# Detector: Page-Hinkley em cima da "surpresa" (loss por passo)
# ============================================================
class PageHinkley:
    def __init__(self, delta=0.001, threshold=7.0, alpha=0.99):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.t = 0

    def update(self, x):
        x = float(x)
        self.t += 1

        if self.t == 1:
            self.mean = x
        else:
            self.mean = self.alpha * self.mean + (1 - self.alpha) * x

        self.cum += (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)

        return (self.cum - self.min_cum) > self.threshold


# -----------------------------
# Modelo
# -----------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(CONTEXT_DIM, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, ARMS)
)


# ============================================================
# Treino para mudança moderada
# - eps piso sempre
# - boost quando detector dispara
# - reset forte só se detecções forem frequentes (mudança "quase hard")
# ============================================================
def train_moderate_change(
    env,
    epochs=40000,
    lr=1e-3,
    batch_size=64,
    recent_window=900,

    eps_base=0.18,        # exploração mínima permanente
    boost_eps=0.65,       # exploração extra após detecção
    boost_steps=450,

    ph_delta=0.001,
    ph_threshold=7.0,
    ph_alpha=0.99,

    # reset forte opcional (se detectar demais em pouco tempo)
    reset_if_detections_in_last=8,
    reset_window=2500,

    print_every=1000
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = deque(maxlen=recent_window)
    rewards = []
    loss_hist = []

    detector = PageHinkley(delta=ph_delta, threshold=ph_threshold, alpha=ph_alpha)

    boost_left = 0
    detections = 0
    det_steps = deque(maxlen=reset_if_detections_in_last)

    for i in range(epochs):
        s = env.get_state()
        state_t = torch.from_numpy(s)

        model.eval()
        with torch.no_grad():
            logits = model(state_t)

        eps = boost_eps if boost_left > 0 else eps_base

        # epsilon-greedy com piso
        if np.random.rand() < eps:
            a = np.random.randint(env.arms)
        else:
            a = int(torch.argmax(logits).item())

        r = env.step(a, s)
        rewards.append(r)
        buffer.append((s, a, r))

        # surpresa do modelo no braço escolhido
        with torch.no_grad():
            loss_step = F.binary_cross_entropy_with_logits(
                logits[a], torch.tensor(float(r))
            ).item()
        loss_hist.append(loss_step)

        # detecção
        if detector.update(loss_step):
            detections += 1
            boost_left = boost_steps
            det_steps.append(i)
            detector.reset()  # evita cascata

        if boost_left > 0:
            boost_left -= 1

        # reset forte só se estiver detectando demais (mudança muito grande/rápida)
        if len(det_steps) == det_steps.maxlen and (det_steps[-1] - det_steps[0]) < reset_window:
            buffer.clear()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            det_steps.clear()

        if len(buffer) < batch_size:
            continue

        # treino batch
        model.train()
        batch = random.sample(list(buffer), batch_size)

        states_b = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        actions_b = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards_b = torch.tensor([x[2] for x in batch], dtype=torch.float32)

        optimizer.zero_grad()
        all_logits = model(states_b)  # (B, ARMS)
        chosen_logits = all_logits.gather(1, actions_b.view(-1, 1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(chosen_logits, rewards_b)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if i % print_every == 0 and i > 0:
            print(
                f"Step {i:5d} | eps={'BOOST' if boost_left>0 else 'BASE '}={eps:.2f} | "
                f"Reward (últimos 1000): {np.mean(rewards[-1000:]):.3f} | "
                f"Loss_step (últimos 1000): {np.mean(loss_hist[-1000:]):.3f} | "
                f"detecções={detections}"
            )

    return np.array(rewards)


# -----------------------------
# Testes top-1 e top-2
# -----------------------------
def test_top1(model, env, num_tests=20):
    print("\n--- TESTE TOP-1 ---")
    model.eval()
    with torch.no_grad():
        hits = 0
        for i in range(num_tests):
            s = env.get_state()
            logits = model(torch.from_numpy(s))
            pred = int(torch.argmax(logits).item())

            real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
            best = int(np.argmax(real_scores))

            ok = (pred == best)
            hits += int(ok)
            print(f"Teste {i+1:2d}: escolha={pred} melhor_real={best} {'✅' if ok else '❌'}")
        print(f"Acertos: {hits}/{num_tests}")


def test_topk(model, env, num_tests=20, k=2):
    print(f"\n--- TESTE TOP-{k} ---")
    model.eval()
    with torch.no_grad():
        hits = 0
        for i in range(num_tests):
            s = env.get_state()
            logits = model(torch.from_numpy(s)).numpy()
            topk = [int(x) for x in np.argsort(logits)[-k:][::-1]]

            real_scores = np.array([np.dot(s, env.hidden_weights[:, a]) for a in range(env.arms)])
            best = int(np.argmax(real_scores))

            ok = best in topk
            hits += int(ok)
            print(f"Teste {i+1:2d}: top{k}={topk} melhor_real={best} {'✅' if ok else '❌'}")
        print(f"Acertos: {hits}/{num_tests}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    env = RichContextBanditModerateChange(
        arms=ARMS,
        context_dim=CONTEXT_DIM,
        min_change=600,
        max_change=1800,
        mix_low=0.25,
        mix_high=0.55,
        noise_sigma=0.02
    )

    print("Treino (mudança moderada em blocos + detecção por loss_step)...")
    train_moderate_change(
        env,
        epochs=40000,
        lr=1e-3,
        batch_size=64,
        recent_window=900,
        eps_base=0.18,
        boost_eps=0.65,
        boost_steps=450,
        ph_delta=0.001,
        ph_threshold=7.0,
        ph_alpha=0.99,
        print_every=1000
    )

    test_top1(model, env, num_tests=20)
    test_topk(model, env, num_tests=20, k=2)




# -----------------------------
# Mudança nao estacionaro a cada 500
# -----------------------------
# if __name__ == "__main__":
#     np.random.seed(0)
#     torch.manual_seed(0)
#     random.seed(0)

#     env = RichContextBanditNonStationary(
#         arms=ARMS,
#         context_dim=CONTEXT_DIM,
#         update_every=1000
#     )

#     print("Iniciando treino (NÃO-ESTACIONÁRIO) com janela curta + ε com piso + boost após mudança...")
#     rewards_history = train_nonstationary(
#         env,
#         epochs=30000,
#         lr=1e-3,
#         batch_size=64,
#         recent_window=400,
#         eps_start=0.5,
#         eps_end=0.15,
#         eps_decay_steps=15000,
#         warmup_steps=2000,
#         boost_eps=0.6,
#         boost_steps=250,
#         print_every=1000
#     )

#     # Teste estilo "acerta/erra"
#     test_top1(model, env, num_tests=20)
#     test_topk(model, env, num_tests=20, k=2)

#     print("\nBandit Hidden Weights (mundo atual):")
#     print(np.round(env.hidden_weights, 2))




# -----------------------------
# Main MUDANÇA DE AMBIENTE ESTACIOARIO
# -----------------------------
# if __name__ == "__main__":
#     # seeds (reprodutível)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     random.seed(0)

#     env = RichContextBanditStationary(arms=ARMS, context_dim=CONTEXT_DIM)

#     print("Iniciando treino (ESTACIONÁRIO) com BCE (logits) + epsilon decay...")
#     rewards_history = train_stationary(
#         env,
#         epochs=30000,
#         lr=1e-3,
#         batch_size=64,
#         recent_window=5000,
#         eps_start=0.5,
#         eps_end=0.05,
#         eps_decay_steps=15000,
#         warmup_steps=2000,
#         print_every=1000
#     )

#     # Testes estilo "acerta/erra"
#     test_top1(model, env, num_tests=12)

#     # (Opcional) mais estável
#     test_topk(model, env, num_tests=12, k=3)

#     print("\nBandit Hidden Weights (lógica oculta do mundo):")
#     print(np.round(env.hidden_weights, 2))

    
# if __name__ == "__main__":
#     env = ContextBandit(10,update_every=1000)
#     reward = train(env, 10)
#     print_policy_per_state(model, arms=10, tau=2.0)
#     print("\nBandit matrix (probabilidades):")
#     print(env.bandit_matrix)
#     true_best = env.bandit_matrix.argmax(axis=1)
#     print("true best arms:", true_best)