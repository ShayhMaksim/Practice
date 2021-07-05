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

zone1=BadZone(10,10,20,15)
zone2=BadZone(5,5,7,15)
Map.AddWalls([zone1,zone2])
game=Map.reset()

app = QApplication(sys.argv)

n_episode=5000
#q_learning
# gamma=1
# alpha=0.3
# epsilon=0.1

#sarsa
epsilon=0.03
alpha=0.4
gamma=1

length_episode=[0] * n_episode
total_reward_episode=[0] * n_episode

def gen_random_policy(n_action):
    probs = torch.ones(n_action) / n_action
    def policy_function(state):
        return probs
    return policy_function

random_policy = gen_random_policy(game.player.action_space)

def run_episode(env, behavior_policy):
    """
    Выполняет один эпизод, следуя заданной поведенческой стратегии
    @param env: имя окружающей среды OpenAI Gym
    @param behavior_policy: поведенческая стратегия
    @return: результирующие состояния, действия и вознаграждения для всего эпизода
    """
    env = Map.reset()
    state=game.posPlayer()
    rewards = []
    actions = []
    states = []
    is_done = False
    while not is_done:
        probs = behavior_policy(state)
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done = env.step(action)
        rewards.append(reward)
        if is_done:
            break
    return states, actions, rewards


"""Определяем жадную стратегию"""
def gen_policy():
    def policy_function(state,Q):
        best_action = torch.argmax(Q[state]).item()      
        return best_action
    return policy_function


epsilon_greedy_policy=gen_policy()

from collections import defaultdict
def mc_control_off_policy_weighted(env, gamma, n_episode,behavior_policy):
    """
    Строит оптимальную стратегию методом управления МК
    с разделенной стратегией и взвешенной выборкой по значимости
    @param env: имя окружающей среды OpenAI Gym
    @param gamma: коэффициент обесценивания
    @param n_episode: количество эпизодов
    @param behavior_policy: поведенческая стратегия
    @return: оптимальные Q-функция и стратегия
    """
    n_action = env.player.action_space
    N = defaultdict(float)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        W = 1.
        states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
        return_t = 0.
        for state_t, action_t, reward_t in zip(states_t[::-1],actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            N[(state_t, action_t)] += W
            Q[state_t][action_t] += (W / N[(state_t, action_t)]) * (return_t - Q[state_t][action_t])
            if action_t != torch.argmax(Q[state_t]).item():
                break
            W *= 1./ behavior_policy(state_t)[action_t]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy

gamma = 1
n_episode = 100

optimal_Q, optimal_policy = mc_control_off_policy_weighted(game,gamma, n_episode, random_policy)

game=Map(3,3,27,27)
state=game.posPlayer()
#env.render()
is_done=False
while not is_done:
    action=epsilon_greedy_policy(state,optimal_Q)
    next_state,reward,is_done=game.step(action)
    state=next_state
    #env.render()


import matplotlib.pyplot as plt
plt.plot(length_episode)
plt.title('Зависимсоть длины эпизода от времени')
plt.xlabel('Эпизод')
plt.ylabel('Длина')
plt.show()

plt.plot(total_reward_episode)
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()


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
        if game.chart[i][j]==0:#неразведованные зоны
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,255))
        j=j+1
    i=i+1
    j=0


sys.exit(app.exec())

