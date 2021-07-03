import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ActorCriticModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ActorCriticModel, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.action = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)
    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)
        return action_probs, state_values

    
class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def predict(self, s):
        """
        Вычисляет выход, применяя модель исполнитель–критик
        @param s: входное состояние
        @return: вероятности действий, ценность состояния
        """
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs, state_values):
        """
        Обновляет веса сети исполнитель–критик на основе переданных
        обучающих примеров
        @param returns: доход (накопительное вознаграждение) на
                        каждом шаге эпизода
        @param log_probs: логарифм вероятности на каждом шаге
        @param state_values: ценности состояний на каждом шаге
        """
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value, Gt)
            loss += policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
        Предсказывает стратегию, выбирает действие и вычисляет логарифм
        его вероятности
        @param s: входное состояние
        @return: выбранное действие и логарифм его вероятности
        """
        action_probs, state_value = self.predict(s)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob, state_value


def actor_critic(env, estimator, n_episode, gamma=1.0):
    """
    Алгоритм исполнитель–критик
    @param env: имя окружающей среды Gym
    @param estimator: сеть стратегии
    @param n_episode: количество эпизодов
    @param gamma: коэффициент обесценивания
    """
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        env=Map.reset()
        state=env.posPlayer()
        while True:
            one_hot_state = [0]*env.observation
            one_hot_state[state] = 1
            action, log_prob, state_value = estimator.get_action(one_hot_state)
            next_state, reward, is_done = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)
            if is_done:
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs, state_values)
                print('Эпизод: {}, полное вознаграждение: {}'.format(episode, total_reward_episode[episode]))
                if total_reward_episode[episode] >= 450:
                    estimator.scheduler.step()
                break
            state = next_state

n_state = game.observation
n_action = game.player.action_space
n_hidden = 128

lr = 0.03
policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

gamma = 0.9

n_episode = 1500
total_reward_episode = [0] * n_episode
actor_critic(game, policy_net, n_episode, gamma)

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