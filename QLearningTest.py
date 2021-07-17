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

app = QApplication(sys.argv)

zone1=BadZone(10,10,25,12)
zone2=BadZone(5,5,7,25)
zone3=BadZone(23,20,25,28)

# bonus1=Bonus(15,15,16,16)
# bonus2=Bonus(15,5,16,6)
# bonus3=Bonus(20,5,22,6)


# Map.AddBonus([bonus2,bonus3])

Map.AddWalls([zone1,zone2,zone3])

"""Определяем жадную стратегию"""
def gen_epsilon_greedy_policy(n_action,epsilon):
    def policy_function(state,Q):
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

game=Map(3,3,27,27)
epsilon=0.0
epsilon_greedy_policy=gen_epsilon_greedy_policy(game.player.action_space,epsilon)

optimal_Q = torch.load('model/q_learning10')
optimal_policy = torch.load('model/optimal_policy10')

game=Map(3,3,27,27)
state=game.posPlayer()
total_reward=0
#env.render()
is_done=False
while not is_done:
    action=epsilon_greedy_policy(state,optimal_Q)
    next_state,reward,is_done=game.step(action)
    state=next_state
    total_reward+=reward
    #env.render()
print('Полное вознаграждение: {}'.format(total_reward))
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