from A2C import PolicyNetwork
from A2C import ActorCriticModel
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


app = QApplication(sys.argv)

zone1=BadZone(10,10,25,12)
zone2=BadZone(5,5,7,25)
zone3=BadZone(23,20,25,28)


Map.AddWalls([zone1,zone2,zone3])
game=Map.reset()

n_state = game.observation
n_action = game.player.action_space
n_hidden = 128

lr = 0.01
policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)


model=torch.load('model/A2C2')

policy_net.model=model

game=Map(3,3,27,27)
state=game.posPlayer()
#env.render()
total_reward=0
is_done=False
while not is_done:
    one_hot_state = [0]*game.observation
    one_hot_state[state] = 1
    action, log_prob, state_value = policy_net.get_action(one_hot_state)
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
        if game.chart[i][j]==0:#неразведованные зоны
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,255))
        j=j+1
    i=i+1
    j=0


sys.exit(app.exec())