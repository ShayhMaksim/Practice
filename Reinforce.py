import gym
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)

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


zone1=BadZone(10,10,25,12)
zone2=BadZone(5,5,7,25)
zone3=BadZone(23,20,25,28)


Map.AddWalls([zone1,zone2,zone3])
game=Map.reset()

app = QApplication(sys.argv)


class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        

    def predict(self, s):
        """
        Вычисляет вероятности действий в состоянии s,
        применяя обученную модель
        @param s: входное состояние
        @return: предсказанная стратегия
        """
        return self.model(torch.Tensor(s))

    def update(self, advantages, log_probs):
        """
        Обновляет веса сети стратегии на основе обучающих примеров
        @param advantages: преимущества на каждом шаге эпизода
        @param log_probs: логарифмы вероятностей на каждом шаге
        """
        policy_gradient = []
        main_loss=None
        
        for log_prob, Gt in zip(log_probs, advantages):
            policy_gradient.append(-log_prob * Gt)

            loss = torch.stack(policy_gradient).sum()
            if main_loss==None:
                main_loss=loss
            else:
                main_loss+=loss
        
        
        self.optimizer.zero_grad()         
        main_loss.backward(retain_graph=True)
        self.optimizer.step()
     

            


    def get_action(self, s):
        """
        Предсказывает стратегию, выбирает действие и вычисляет логарифм
        его вероятности
        @param s: входное состояние
        @return: выбранное действие и логарифм его вероятности
        """
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob



class ValueNetwork():
    def __init__(self, n_state, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
        torch.nn.Linear(n_state, n_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        """
        Обновить веса DQN, получив обучающий пример
        @param s: состояния
        @param y: целевые ценности
        """
        y_pred = self.model(torch.Tensor(s))
        new_shape = (len(y_pred), 1)
        
        loss = self.criterion(y_pred, Variable(torch.Tensor(y.view(new_shape))))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def predict(self, s):
        """
        Вычисляет значения Q-функции состояния для всех действий,
        применяя обученную модель
        @param s: входное состояние
        @return: значения Q-функции состояния для всех действий
        """
        with torch.no_grad():
            return self.model(torch.Tensor(s))

def reinforce(env, estimator_policy, estimator_value,n_episode, gamma=1.0):
    """
    Алгоритм REINFORCE с базой
    @param env: имя окружающей среды Gym
    @param estimator: сеть стратегии
    @param estimator_value: сеть ценности
    @param n_episode: количество эпизодов
    @param gamma: коэффициент обесценивания
    """
    for episode in range(n_episode):
        log_probs = []
        states = []
        rewards = []
        env=Map.reset()
        state=env.posPlayer()
        while True:
            one_hot_state = [0]*env.observation
            one_hot_state[state] = 1
            states.append(one_hot_state)
            action, log_prob = estimator_policy.get_action(one_hot_state)
            next_state, reward, is_done = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:
                Gt = 0
                pw = 0
                returns = []
                for t in range(len(states)-1, -1, -1):
                    Gt += gamma ** pw * rewards[t]
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                baseline_values = estimator_value.predict(states)
                advantages = returns - baseline_values
                estimator_value.update(states, returns)

                estimator_policy.update(advantages, log_probs)

                print('Эпизод: {}, полное вознаграждение: {}'.format(episode, total_reward_episode[episode]))
                break
            state = next_state

n_state = game.observation
n_action = game.player.action_space
n_hidden_p = 8
lr_p = 0.003
policy_net = PolicyNetwork(n_state, n_action, n_hidden_p, lr_p)

n_hidden_v = 8
lr_v = 0.003
value_net = ValueNetwork(n_state, n_hidden_v, lr_v)
gamma = 0.9

n_episode = 500
total_reward_episode = [0] * n_episode
reinforce(game, policy_net, value_net, n_episode, gamma)

import matplotlib.pyplot as plt
plt.plot(total_reward_episode)
plt.title('Зависимость вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()

game=Map(3,3,27,27)
state=game.posPlayer()
#env.render()
is_done=False
while not is_done:
    one_hot_state = [0]*game.observation
    one_hot_state[state] = 1
    action, log_prob, state_value = policy_net.get_action(one_hot_state)
    next_state,reward,is_done=game.step(action)
    state=next_state
    #env.render()

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