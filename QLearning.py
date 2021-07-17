from Bonus import Bonus
from collections import defaultdict
import torch
from Map import Map
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QBrush, QColor
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget,QApplication,QGraphicsView,QGraphicsScene
import sys
from BadZone import BadZone
import pathlib
import json

zone1=BadZone(10,10,25,12)
zone2=BadZone(5,5,7,25)
zone3=BadZone(23,20,25,28)

# bonus1=Bonus(15,15,16,16)
# bonus2=Bonus(15,5,16,6)
# bonus3=Bonus(20,5,21,6)

Map.AddWalls([zone1,zone2,zone3])

# Map.AddBonus([bonus1,bonus2,bonus3])

game=Map.reset()

app = QApplication(sys.argv)

n_episode=7000
#q_learning
gamma=1
alpha=0.3
epsilon=0.9

#sarsa
# epsilon=0.03
# alpha=0.4
# gamma=0.05

length_episode=[0] * n_episode
total_reward_episode=[0] * n_episode

"""Определяем жадную стратегию"""
def gen_epsilon_greedy_policy(n_action):
    def policy_function(state,Q,epsilon):
        # probs = torch.ones(n_action) * epsilon / n_action
        # best_action = torch.argmax(Q[state]).item()
        # probs[best_action] += (1.0 - epsilon)
        # action = torch.multinomial(probs, 1).item()
        # return action
        if np.random.random()<epsilon:
            return np.random.choice(n_action)
        else:
            return torch.argmax(Q[state]).item()
    return policy_function

epsilon_greedy_policy=gen_epsilon_greedy_policy(game.player.action_space)

"""Функция, выполняющая Q-обучение"""
def q_learning(env,gamma,n_episode,alpha,epsilon):
    """
    @param env:имя окрудающей среды OpenAI Gym
    @param gamma: коэффициент обесценивания
    @param n_episode: количество эпизодов
    @return: оптимальные Q-функция и стратегия
    """
    n_action=env.player.action_space

    Q=defaultdict(lambda: torch.zeros(n_action))
    #env.render()
    for episode in range(n_episode):
        env=Map.reset()
        state=env.posPlayer()
        is_done=False 
        while not is_done:
            action=epsilon_greedy_policy(state,Q,epsilon)
            next_state,reward,is_done=env.step(action)
            td_delta=reward+gamma*torch.max(Q[next_state])-Q[state][action]
            Q[state][action]+=alpha*td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            #env.render()
            if is_done:
                if epsilon>0.4:
                    epsilon=epsilon*0.9997
                else:
                    epsilon=0.4
                print('Эпизод: {}, полное вознаграждение: {}'.format(episode, total_reward_episode[episode]))
                break
            state=next_state
    policy = {}
    for state,actions in Q.items():
        policy[state]=torch.argmax(actions).item()
    return Q,policy


def double_q_learning(env, gamma, n_episode, alpha):
    """
    Строит оптимальную стратегию методом двойного Q-обучения с
    разделенной стратегией
    @param env: имя окружающей среды OpenAI Gym
    @param gamma: коэффициент обесценивания
    @param n_episode: количество эпизодов
    @return: оптимальные Q-функция и стратегия
    """
    n_action = env.player.action_space
    n_state=Map.observation
    Q1 = torch.zeros(n_state,n_action)
    Q2 = torch.zeros(n_state,n_action)
    for episode in range(n_episode):
        env=Map.reset()
        state=env.posPlayer()
        is_done=False
        while not is_done:
            action = epsilon_greedy_policy(state, Q1 + Q2)
            next_state, reward, is_done = env.step(action)
            if (torch.rand(1).item() < 0.5):
                best_next_action = torch.argmax(Q1[next_state])
                td_delta = reward + gamma * Q2[next_state][best_next_action]- Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                best_next_action = torch.argmax(Q2[next_state])
                td_delta = reward + gamma * Q1[next_state][best_next_action]- Q2[state][action]
                Q2[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                print('Эпизод: {}, полное вознаграждение: {}'.format(episode, total_reward_episode[episode]))
                break
            state = next_state
    policy = {}
    Q = Q1 + Q2
    for state in range(n_state):
        policy[state] = torch.argmax(Q[state]).item()
    return Q, policy
                

def sarsa(env, gamma, n_episode, alpha,epsilon):
    """
    Строит оптимальную стратегию методом SARSA с единой стратегией
    @param env: имя окружающей среды OpenAI Gym
    @param gamma: коэффициент обесценивания
    @param n_episode: количество эпизодов
    @return: оптимальные Q-функция и стратегия
    """
    n_action = env.player.action_space
    Q = defaultdict(lambda: torch.zeros(n_action))
    for episode in range(n_episode):
        env=Map.reset()
        state=env.posPlayer()
        is_done=False
        action = epsilon_greedy_policy(state, Q,epsilon)
        while not is_done:
            next_state, reward, is_done = env.step(action)
            next_action = epsilon_greedy_policy(next_state, Q,epsilon)
            td_delta = reward + gamma * Q[next_state][next_action]- Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                if epsilon>0.4:
                    epsilon=epsilon*0.9997
                else:
                    epsilon=0.4
                break
            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy




optimal_Q,optimal_policy=sarsa(game,gamma,n_episode,alpha,epsilon)

opt_Q = {}

for key,value in optimal_Q.items():
    opt_Q[key]=value

# torch.save(opt_Q,'model/q_learning10')
# torch.save(optimal_policy,'model/optimal_policy10')

torch.save(opt_Q,'model/sarsa3')
torch.save(optimal_policy,'model/optimal_policy_sarsa3')

# torch.save(optimal_Q,'model/double_q_learning')
# torch.save(optimal_policy,'model/double_optimal_policy')



# alpha_options = [0.3,0.4, 0.5, 0.6]
# epsilon_options = [0.2,0.1, 0.03, 0.01]
# n_episode = 1000

# for alpha in alpha_options:
#     for epsilon in epsilon_options:
#         length_episode = [0] * n_episode
#         total_reward_episode = [0] * n_episode
#         sarsa(game, gamma, n_episode, alpha)
#         reward_per_step = [reward/float(step) for reward, step in zip(total_reward_episode, length_episode)]
#         print('alpha: {}, epsilon: {}'.format(alpha, epsilon))
#         print('Среднее вознаграждение в {} эпизодах: {}'.format(n_episode, sum(total_reward_episode) / n_episode))
#         print('Средняя длина {} эпизодов: {}'.format(n_episode, sum(length_episode) / n_episode))
#         print('Среднее вознаграждение на одном шаге в {} эпизодах:{}\n'.format(n_episode, sum(reward_per_step) / n_episode))

game=Map(3,3,27,27)
state=game.posPlayer()
#env.render()
is_done=False
total_reward=0
while not is_done:
    action=epsilon_greedy_policy(state,optimal_Q,epsilon)
    next_state,reward,is_done=game.step(action)
    total_reward+=reward  
    state=next_state  
    #env.render()


import matplotlib.pyplot as plt
plt.plot(length_episode)
plt.title('Зависимсоть длины эпизода от времени')
plt.xlabel('Эпизод')
plt.ylabel('Длина')
plt.savefig('Зависимсоть длины эпизода от времени s')
plt.show()

plt.plot(total_reward_episode)
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.savefig('Зависимсоть вознаграждения в эпизоде от времени s')
plt.show()

data={'Длина эпизода':length_episode, 'Полное вознаграждение':total_reward_episode}
df = pd.DataFrame(data)
df.to_csv('Sarsa2')

scene = QGraphicsScene()
graphicsView = QGraphicsView(scene)
graphicsView.show()
graphicsView.resize(1000,1000)

i=0
j=0
c=18

while i<30:
    while j<30:
        if game.chart[i][j]==1:#барьер
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,0,255),QBrush(QColor(0,0,255)))
        if game.chart[i][j]==2:#финиш
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,0,0),QBrush(QColor(255,0,0)))
        if game.chart[i][j]==3:#текущее положение игрока
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,0),QBrush(QColor(0,255,0)))
        if game.chart[i][j]==4:#опасные зоны
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,155,150),QBrush(QColor(255,155,150)))
        if game.chart[i][j]==5:#зоны в которых побывали
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,155,150),QBrush(QColor(0,155,150)))
        if game.chart[i][j]==6:#зоны в которых можно получить бонус
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(100,155,150),QBrush(QColor(100,155,0)))
        if game.chart[i][j]==7:#зоны в которых можно получить бонус
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,0,0),QBrush(QColor(0,0,0)))  
        if game.chart[i][j]==0:#неразведованные зоны
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,255))
        j=j+1
    i=i+1
    j=0


sys.exit(app.exec())