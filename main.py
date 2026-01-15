import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n = 20
window_size = 100 # Tamanho da janela (últimos 100 jogos)
current_window = [] 
moving_avg_rewards = [] 
probs = np.random.rand(n) #A
eps = 0.1
#Uma rodada de uma máquina
def get_reward(prob, n=10):
    reward = 0;
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

record = np.zeros((n,2))
record[:, 1] = 10.0
#percorre a matriz record para pegar a melhor maquina
def get_best_arm(record):
    arm_index = np.argmax(record[:,1],axis=0)
    return arm_index


# def update_record(record,action,r):
#     new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1)
#     record[action,0] += 1
#     record[action,1] = new_r
#     return record

def update_record(record, action, r):
    alpha = 0.1  # Taxa de aprendizado (10% de importância para o novo)
    
    # MUDOU AQUI: Em vez da média aritmética, usamos a fórmula do Alpha
    new_r = record[action,1] + alpha * (r - record[action,1])
    
    record[action,0] += 1  # Mantemos o contador apenas para seu controle
    record[action,1] = new_r
    return record


def softmax(av, tau=0.8):
 softm = np.exp(av / tau) / np.sum( np.exp(av / tau) )
 return softm


fig,ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9,5)
rewards = [0]
rec_inicial = 0
#Com 20 máquinas 200 rodadas nao converge
for i in range(4000):
    print('PROBABILIDADE DE CADA %s', probs)
    print(record)
    #epsilon = 1.0 / np.sqrt(i + 1)
    #epsilon = 0.3 * (0.99 ** i) muito agressivo, para de explorar mt cedo
    # if random.random() > 0.3:
    #     choice = get_best_arm(record)
    # else:
    #     choice = np.random.randint(20)
    if i == 2000:
        rec_inicial = mean_reward
        probs = np.random.rand(n)
    #r = get_reward(probs[choice])
    p = softmax(record[:,1])
    choice = np.random.choice(np.arange(n),p=p)
    r = get_reward(probs[choice])
    record = update_record(record,choice,r)
    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)
    rewards.append(mean_reward)
    # Este reward n converge 100% para o valor da maquina, pois a funçao
    # tem memória das recompensas passadas ainda da explorçao, tb pq
    # ele explora até o fim ainda. Aqui não eh um problema, mas se 
    # o ambiente for estacionário, ele será.
    # Ex, no inicio uma máquina era muito boa, e terá recompensas altas
    # deopis ele pode ter baixa, mas o passado ainda terá um peso q levanta.
    print('REWARD_INICIAL ',rec_inicial)
    print('REWARD ',mean_reward)
    # 2. CÁLCULO DA MÉDIA MÓVEL (O novo)
    current_window.append(r)
    if len(current_window) > window_size:
        current_window.pop(0) # Remove o mais velho
    moving_avg_rewards.append(np.mean(current_window))
ax.plot(moving_avg_rewards, label=f'Média Móvel (Últimos {window_size})', color='orange', alpha=0.8)
ax.plot(np.arange(len(rewards)),rewards)
plt.show()


#ALTERANDO A FUNÇAO PROB NO MEIO DO EPISÓDIO, O ALGORITMO NAO CONVERGE.
# mesmo com maior peso para recompensas atuais, nao converge

#SOLUÇÔES?
# Detectou que a recompensa média global caiu muito? (Isso indica que a 
# máquina boa quebrou). Automaticamente suba o Epsilon para 1.0 (100%) por algumas rodadas. 
# Isso força o agente a re-testar todo mundo, dando chance para a máquina B mostrar seu 
# novo valor

# Ao aumentar o Alfa (ex: para 0.4 ou 0.5), você resolveu o problema do "atraso" (o agente
# percebe rápido a mudança). 
# Mas você criou um novo problema: a Instabilidade.
# Imagine que você tem uma Máquina Boa que paga prêmio 80% das vezes. 
# Mas, como é um jogo de azar, às vezes ela não paga nada (tem azar).
# A Consequência: Por causa de um único azar, 
# o agente derrubou a nota da máquina de "Excelente" (0.8) para "Ruim" (0.4)
# Se o seu ambiente muda muito rápido (ex: a cada 50 jogadas tudo muda), você é obrigado a aceitar 
# o risco 
# e usar um Alfa alto. Se ele muda raramente (a cada 1000 jogadas), um Alfa baixo é mais seguro


# Softmax é uma alternativa para melhorar a exploraçao, mas tb é projudicada se for nao estac. 
# ja que o agente ira ter probabilidades mais altas para as maquinas antigas. Mesmo que haja a 
# probablidade de testar todas, se a boa for muito baixa, demoraria muito.
# uma solução seria "reaquecimento do tal" ao ver que a recompensa variou muito.
# Porem a recompensa é a media da recompensa até agora, entao tem muito peso da recompensa
# ja convergida, entao a mudança brusca nao causaria variaçao.
# usar média movel? a media normal até que funciona


#OBS, maneira de cotornar Convergência Prematura (O Fenômeno)
#É quando o algoritmo de aprendizado "acha" que já aprendeu tudo o que
#precisava e estabiliza, mas ele parou de aprender antes de encontrar a melhor solução real.
# o softmax pode aprender errado com uma boa eh ruim e dar uma probabilidade ruim pra ela
# SOLUÇÃO: começar otimista: A lógica inverte o jogo: em vez de a 
# máquina ter que provar que é boa para ser escolhida, ela tem que provar que é ruim