from typing import List
from PyQt5.QtWidgets import QGraphicsScale, QGraphicsScene, QGraphicsView
from Player import Player
from Finish import Finish
from BadZone import BadZone
import numpy as np
import random
#класс карты
class Map:
    
    def __init__(self,x0,y0,xf,yf) -> None:
        self.chart=np.zeros((50,50))

        for i in range(0,50):
            self.chart[0][i]=1
            self.chart[49][i]=1
            self.chart[i][49]=1
            self.chart[i][0]=1


        self.player=Player(x0,y0,self.chart)
        self.finish=Finish(xf,yf)

        self.chart[x0][y0]=self.player.symbol
        self.chart[xf][yf]=self.finish.symbol

    def AddWalls(self,zones:List[BadZone]):
        self.zones=zones
        for zone in zones:
            for i in range(zone.x0,zone.x):
                for j in range(zone.y0,zone.y):
                    self.chart[i][j]=zone.symbol

    @classmethod 
    def reset(self):
        return Map(10,10,45,45)
    
    def posPlayer(self):
        return self.encode(self.player.x,self.player.y)

    def step(self,command):
        self.player.Move(command)
        return self.encode(self.player.x,self.player.y),self.player.score,self.player.isDone


    def encode(self,player_x,player_y):
        i = player_x
        i *= 50
        i += player_y
        return i

    
    def decode(self,i):
        out=[]
        out.append(i%50)
        i=i // 50
        out.append(i)
        assert(0<=i<50)
        return reversed(out)