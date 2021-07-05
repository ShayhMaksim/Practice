import gym
import torch
from torch.autograd import Variable
import random
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

class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        """
        Обновляет веса DQN, получив обучающий пример
        @param s: состояние
        @param y: целевое значение
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        """
        Вычисляет значения Q-функции состояния для всех действий,
        применяя обученную модель
        @param s: входное состояние
        @return: значения Q для всех действий
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
        return torch.argmax(q_values).item()
    return policy_function


def q_learning(env, estimator, n_episode, gamma=1.0,epsilon=0.1, epsilon_decay=.99):
    """
    Глубокое Q-обучение с применением DQN
    @param env: имя окружающей среды Gym
    @param estimator: объект класса Estimator
    @param n_episode: количество эпизодов
    @param gamma: коэффициент обесценивания
    @param epsilon: параметр ε-жад­ной стратегии
    @param epsilon_decay: коэффициент затухания epsilon
    """
    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        env=Map.reset()
        state=env.posPlayer()
        is_done = False
        while not is_done:
            one_hot_state = [0]*env.observation
            one_hot_state[state] = 1
            action = policy(one_hot_state)
            
            next_state, reward, is_done = env.step(action)
            total_reward_episode[episode] += reward
            #modified_reward = next_state + 0.5
            # if next_state[0] >= 0.5:
            #     modified_reward += 100
            # elif next_state[0] >= 0.25:
            #     modified_reward += 20
            # elif next_state[0] >= 0.1:
            #     modified_reward += 10
            # elif next_state[0] >= 0:
            #     modified_reward += 5
            q_values = estimator.predict(one_hot_state).tolist()
            if is_done:
                q_values[action] = reward
                estimator.update(one_hot_state, q_values)
                break
            next_one_hot_state = [0]*env.observation
            next_one_hot_state[next_state] = 1
            q_values_next = estimator.predict(next_one_hot_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            estimator.update(one_hot_state, q_values)
            state = next_state
        print('Эпизод: {}, полное вознаграждение: {}, epsilon:{}'.format(episode,total_reward_episode[episode], epsilon))
        epsilon = max(epsilon * epsilon_decay, 0.01)

n_state = game.observation
n_action = game.player.action_space
n_hidden = 64

lr = 0.001
dqn = DQN(n_state, n_action, n_hidden, lr)

n_episode = 1000
total_reward_episode = [0] * n_episode
q_learning(game, dqn, n_episode, gamma=.99, epsilon=.3)

import matplotlib.pyplot as plt

game=Map(3,3,27,27)
state=game.posPlayer()
#env.render()
is_done=False
while not is_done:
    one_hot_state = [0]*game.observation
    one_hot_state[state] = 1
    action, log_prob, state_value = gen_epsilon_greedy_policy(dqn, .3, n_action)
    next_state,reward,is_done=game.step(action)
    state=next_state
    #env.render()


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