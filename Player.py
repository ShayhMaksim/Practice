import random 
from Finish import Finish
class Player:
    '''
        Основные команды актера:
        8: совершить посадку и выгрузить груз (убрали)
        0: север
        1: юг
        2: запад
        3: восток
        4: северо-запад
        5: северо-восток
        6: юго-запад
        7: юго-восток
    '''
    symbol=3
    action_space=8
    def __init__(self,x,y,map,finish:Finish) -> None:
        self.x=x
        self.y=y
        self.map=map
        self.score=0
        self.isDone=False
        self.finish=finish
        self.normalize=self.Distance()

    def Distance(self):
        return ((self.x-self.finish.x)**2+(self.y-self.finish.y)**2)**0.5

    def Move(self,command):
        # if command==9:
        #     if self.map[self.x][self.y]==2:
        #         self.score=500
        #         self.isDone=True
        #     else:
        #         self.score=-250
        #     return
        penalty0=-1
        penalty4=-500
        penalty5=-200
        bonus=500
        if command==0:
            if self.map[self.x-1][self.y]!=1:
                if self.map[self.x-1][self.y]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x-1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==1:
            if self.map[self.x+1][self.y]!=1:
                if self.map[self.x+1][self.y]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x+1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==2:
            if self.map[self.x][self.y-1]!=1:
                if self.map[self.x][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x][self.y-1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.y=self.y-1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==3:
            if self.map[self.x][self.y+1]!=1:
                if self.map[self.x][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x][self.y+1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.y=self.y+1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==4:
            if self.map[self.x-1][self.y-1]!=1:
                if self.map[self.x-1][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y-1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x-1
                self.y=self.y-1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==5:
            if self.map[self.x-1][self.y+1]!=1:
                if self.map[self.x-1][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y+1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x-1
                self.y=self.y+1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==6:
            if self.map[self.x+1][self.y-1]!=1:
                if self.map[self.x+1][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y-1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x+1
                self.y=self.y-1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        if command==7:
            if self.map[self.x+1][self.y+1]!=1:
                if self.map[self.x+1][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y+1]==5:
                    self.score=penalty5
                else:
                    self.score=penalty0
                self.map[self.x][self.y]=5
                self.x=self.x+1
                self.y=self.y+1
                #self.score=-1
                self.score=self.score*self.Distance()/self.normalize
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                self.map[self.x][self.y]=self.symbol
        
        