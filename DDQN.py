import gym
import torch
from torch.autograd import Variable
import random
from collections import defaultdict, deque
import torch
from Map import Map
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QBrush, QColor
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget,QApplication,QGraphicsView,QGraphicsScene
import sys
from BadZone import BadZone
import copy 

zone1=BadZone(10,10,25,12)
zone2=BadZone(5,5,7,25)
zone3=BadZone(23,20,25,28)

Map.AddWalls([zone1,zone2,zone3])
env=Map.reset()

app = QApplication(sys.argv)

n_episode=10000
#q_learning
gamma=0.95
alpha=0.3
epsilon=0.1



class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model_target = copy.deepcopy(self.model)

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

    def target_predict(self, s):
        """
        Вычисляет значения Q-функции состояния для всех действий
        с помощью целевой сети
        @param s: входное состояние
        @return: целевые ценности состояния для всех действий
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def replay(self, memory, replay_size, gamma):
        """
        Буфер воспроизведения совместно с целевой сетью
        @param memory: буфер воспроизведения опыта
        @param replay_size: сколько примеров использовать при каждом
              обновлении модели
        @param gamma: коэффициент обесценивания
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma *torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
        return torch.argmax(q_values).item()
    return policy_function

def q_learning(env, estimator, n_episode, replay_size,target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    """
    Глубокое Q-обучение методом Double DQN с воспроизведением опыта
    @param env: имя окружающей среды Gym
    @param estimator: объект класса DQN
    @param replay_size: сколько примеров использовать при каждом
              обновлении модели
    @param target_update: через сколько эпизодов обновлять целевую сеть
    @param n_episode: количество эпизодов
    @param gamma: коэффициент обесценивания
    @param epsilon: параметр ε-жад­ной стратегии
    @param epsilon_decay: коэффициент затухания epsilon
    """
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)

        env=Map.reset()
        state=env.posPlayer()
        is_done = False


        while not is_done:
            one_hot_state = [0]*env.observation
            one_hot_state[state] = 1

            action = policy(one_hot_state)
            next_state, reward, is_done  = env.step(action)
            total_reward_episode[episode] += reward

            next_one_hot_state = [0]*env.observation
            next_one_hot_state[next_state] = 1

            memory.append((one_hot_state, action, next_one_hot_state, reward, is_done))
            if is_done:
                break
            estimator.replay(memory, replay_size, gamma)
            state = next_state
        epsilon = max(epsilon * epsilon_decay, 0.01)

n_state = env.observation
n_action = env.player.action_space
n_episode = 600
last_episode = 200

n_hidden_options = [30, 40]
lr_options = [0.001, 0.003]
replay_size_options = [20, 25]
target_update_options = [30, 35]


for n_hidden in n_hidden_options:
    for lr in lr_options:
        for replay_size in replay_size_options:
            for target_update in target_update_options:
                
                random.seed(1)
                torch.manual_seed(1)
                dqn = DQN(n_state, n_action, n_hidden, lr)
                memory = deque(maxlen=10000)
                total_reward_episode = [0] * n_episode
                q_learning(env, dqn, n_episode, replay_size,target_update, gamma=.9, epsilon=1)
                print(n_hidden, lr, replay_size, target_update,
                    sum(total_reward_episode[-last_episode:])/last_episode)