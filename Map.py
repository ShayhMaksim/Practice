from typing import List
from PyQt5.QtWidgets import QGraphicsScale, QGraphicsScene, QGraphicsView
from Player import Player
from Finish import Finish
from BadZone import BadZone
from Bonus import Bonus
import numpy as np
import random
#класс карты
class Map:
    zones: List[BadZone]=[]
    bonuses: List[Bonus]=[]
    observation=30*30#кол-во исходов
    def __init__(self,x0,y0,xf,yf) -> None:
        self.chart=np.zeros((30,30))

        if self.zones!=[]:
            for zone in Map.zones:
                for i in range(zone.x0,zone.x):
                    for j in range(zone.y0,zone.y):
                        self.chart[i][j]=zone.symbol

        if self.bonuses!=[]:
            for bonus in Map.bonuses:
                for i in range(bonus.x0,bonus.x):
                    for j in range(bonus.y0,bonus.y):
                        self.chart[i][j]=bonus.symbol

        for i in range(0,30):
            self.chart[0][i]=1
            self.chart[29][i]=1
            self.chart[i][29]=1
            self.chart[i][0]=1

        self.finish=Finish(xf,yf)
        self.player=Player(x0,y0,self.chart,self.finish)
        

        self.chart[x0][y0]=self.player.symbol
        self.chart[xf][yf]=self.finish.symbol

    @classmethod
    def AddWalls(self,zones:List[BadZone]):
        self.zones=zones

    @classmethod
    def AddBonus(self,bonuses :List[Bonus]):
        self.bonuses=bonuses   

    @classmethod 
    def reset(self):
        #Map(random.randint(2,15),random.randint(2,15),45,45)
        x=random.randint(1,27)
        y=random.randint(1,25)
        
        # zone1=BadZone(10,10,20,15)
        # zone2=BadZone(5,5,7,15)
        # zone3=BadZone(23,20,25,28)
        if 10<=x<=20 and 10<=y<=15:
            x=3
            y=3

        if 5<=x<=7 and 5<=y<=15:
            x=3
            y=3

        if 23<=x<=25 and 20<=y<=28:
            x=3
            y=3

        return Map(3,3,27,27)#Map(5,5,45,45)
    
    def posPlayer(self):
        return self.encode(self.player.x,self.player.y)

    def step(self,command):
        self.player.Move(command)
        return self.encode(self.player.x,self.player.y),self.player.score,self.player.isDone


    def encode(self,player_x,player_y):
        i = player_x
        i *= 30
        i += player_y
        return i

    
    def decode(self,i):
        out=[]
        out.append(i%30)
        i=i // 30
        out.append(i)
        assert(0<=i<30)
        return reversed(out)