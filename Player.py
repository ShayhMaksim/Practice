import random

from Finish import Finish
class Player:
    '''
        Основные команды актера:
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
        self.x0=x
        self.y0=y
        self.normalize=self.Distance()
        #max(abs(self.x-self.finish.x),abs(self.y-self.finish.y))

    def Distance(self):
        return ((self.x-self.finish.x)**2+(self.y-self.finish.y)**2)**0.5

    def Move(self,command):
        penalty0=-1
        penalty4=-500
        penalty5=10
        bonus=self.normalize
        c=2**0.5
        if command==0:
            if self.map[self.x-1][self.y]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x-1][self.y]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0
                self.x=self.x-1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==1:
            if self.map[self.x+1][self.y]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x+1][self.y]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0
                self.x=self.x+1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==2:
            if self.map[self.x][self.y-1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x][self.y-1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0
                self.y=self.y-1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==3:
            if self.map[self.x][self.y+1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x][self.y+1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0
                
                self.y=self.y+1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==4:
            if self.map[self.x-1][self.y-1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x-1][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y-1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0*c
                
                self.x=self.x-1
                self.y=self.y-1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==5:
            if self.map[self.x-1][self.y+1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x-1][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x-1][self.y+1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0*c
                
                self.x=self.x-1
                self.y=self.y+1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==6:
            if self.map[self.x+1][self.y-1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x+1][self.y-1]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y-1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0*c
                
                self.x=self.x+1
                self.y=self.y-1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        if command==7:
            if self.map[self.x+1][self.y+1]!=1:
                if self.map[self.x][self.y]!=7:
                    self.map[self.x][self.y]=5
                if self.map[self.x+1][self.y+1]==4:
                    self.score=penalty4
                elif self.map[self.x+1][self.y+1]==6:
                    self.score=penalty5
                    self.map[self.x][self.y]=7
                else:
                    self.score=penalty0*c
                
                self.x=self.x+1
                self.y=self.y+1
                #self.score=-1
                #self.score=self.score*self.Distance()/self.normalize
                # if self.map[self.x][self.y]==6:#finish
                #     self.isDone=True
                #     self.score=bonus
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=bonus
                    #self.map[self.x0][self.y0]=6
                self.map[self.x][self.y]=self.symbol
        
        