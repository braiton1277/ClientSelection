import random

import matplotlib.pyplot as plt
import numpy as np

#GRÁFICO Q É POSSIVEL OBSERVAR A QUEDA DA RECOMPENSA QUE USA A MÉDIA MOVEL,
# O AGENTE COSTUMA CONVERGIR PARA A RECOMPENSA DA MELHOR MÁQUINA, COMO HÁ NAO ESTACION
# OCORRE UM DESVIO CLARO. (gráficos q nao desviam sao pq a mudança n foi tao drastica nas prob)

# --- CONFIGURAÇÕES ---
n = 20
probs = np.random.rand(n) 
record = np.zeros((n, 2)) 
record[:, 1] = 10.0
rewards = [0] # Para a Média Acumulada

# NOVAS VARIÁVEIS PARA MÉDIA MÓVEL
window_size = 100 # Tamanho da janela (últimos 100 jogos)
current_window = [] 
moving_avg_rewards = [] 

def softmax(av, tau=0.8):
 softm = np.exp(av / tau) / np.sum( np.exp(av / tau) )
 return softm

# Funções simuladas (para o código rodar aqui)
def get_best_arm(record): return np.argmax(record[:, 1])
def get_reward(prob): return 1 if random.random() < prob else 0
def update_record(record, action, r):
    alpha = 0.1
    record[action, 1] += alpha * (r - record[action, 1])
    return record

# Configura o gráfico antes
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(10, 6)

# --- LOOP PRINCIPAL ---
for i in range(3000):
    # Epsilon-Greedy (Mantive o seu)
    # if random.random() > 0.2: # Botei 0.1 pra ser menos agressivo que 0.3
    #     choice = get_best_arm(record)
    # else:
    #     choice = np.random.randint(n)
        
    p = softmax(record[:,1])
    choice = np.random.choice(np.arange(n),p=p)
    r = get_reward(probs[choice])


    if i == 500:
        probs = np.random.rand(n)
        print("--- MUDANÇA DE AMBIENTE (RODADA 1500) ---")
        
    r = get_reward(probs[choice])
    record = update_record(record, choice, r)
    
    # 1. CÁLCULO DA MÉDIA ACUMULADA (O antigo)
    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)
    rewards.append(mean_reward)
    
    # 2. CÁLCULO DA MÉDIA MÓVEL (O novo)
    current_window.append(r)
    if len(current_window) > window_size:
        current_window.pop(0) # Remove o mais velho
    
    # Calcula a média só dessa janelinha e guarda
    moving_avg_rewards.append(np.mean(current_window))

# --- PLOTAGEM ---

# Plot 1: Média Acumulada (Azul) - A "lenta"
ax.plot(rewards, label='Média Acumulada (Histórico Completo)', color='blue', linewidth=2)

# Plot 2: Média Móvel (Laranja) - A "realista"
# O eixo X precisa ajustar um pouco pq a lista móvel começa do zero
ax.plot(moving_avg_rewards, label=f'Média Móvel (Últimos {window_size})', color='orange', alpha=0.8)

# Linha vermelha pra marcar onde mudou
plt.axvline(x=1500, color='red', linestyle='--', label='Mudança nas Probs')

# Legenda para você saber quem é quem
plt.legend()
plt.title("Comparação: Acumulada vs Móvel em Ambiente Variável")
plt.show()