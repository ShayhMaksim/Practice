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


zone1=BadZone(10,10,20,11)
zone2=BadZone(5,5,7,15)
Map.AddWalls([zone1,zone2])
game=Map.reset()

app = QApplication(sys.argv)

n_episode=10000
gamma=1
alpha=0.3
epsilon=0.1

length_episode=[0] * n_episode
total_reward_episode=[0] * n_episode

"""Определяем жадную стратегию"""
def gen_epsilon_greedy_policy(n_action,epsilon):
    def policy_function(state,Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += (1.0 - epsilon)
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

epsilon_greedy_policy=gen_epsilon_greedy_policy(game.player.action_space,epsilon)

"""Функция, выполняющая Q-обучение"""
def q_learning(env,gamma,n_episode,alpha):
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
            action=epsilon_greedy_policy(state,Q)
            next_state,reward,is_done=env.step(action)
            td_delta=reward+gamma*torch.max(Q[next_state])-Q[state][action]
            Q[state][action]+=alpha*td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            #env.render()
            if is_done:
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
                break
            state = next_state
    policy = {}
    Q = Q1 + Q2
    for state in range(n_state):
        policy[state] = torch.argmax(Q[state]).item()
    return Q, policy
                

optimal_Q,optimal_policy=double_q_learning(game,gamma,n_episode,alpha)

#print(optimal_policy)


env=Map.reset()
state=env.posPlayer()
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