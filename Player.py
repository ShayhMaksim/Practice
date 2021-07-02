import random 

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
    def __init__(self,x,y,map) -> None:
        self.x=x
        self.y=y
        self.map=map
        self.score=0
        self.isDone=False


    def Move(self,command):
        # if command==9:
        #     if self.map[self.x][self.y]==2:
        #         self.score=500
        #         self.isDone=True
        #     else:
        #         self.score=-250
        #     return
        if command==0:
            if self.map[self.x-1][self.y]!=1:
                if self.map[self.x-1][self.y]==4:
                    self.score=-50
                elif self.map[self.x-1][self.y]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x-1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==1:
            if self.map[self.x+1][self.y]!=1:
                if self.map[self.x+1][self.y]==4:
                    self.score=-50
                elif self.map[self.x+1][self.y]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x+1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==2:
            if self.map[self.x][self.y-1]!=1:
                if self.map[self.x][self.y-1]==4:
                    self.score=-50
                elif self.map[self.x][self.y-1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.y=self.y-1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==3:
            if self.map[self.x][self.y+1]!=1:
                if self.map[self.x][self.y+1]==4:
                    self.score=-50
                elif self.map[self.x][self.y+1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.y=self.y+1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==4:
            if self.map[self.x-1][self.y-1]!=1:
                if self.map[self.x-1][self.y-1]==4:
                    self.score=-50
                elif self.map[self.x-1][self.y-1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x-1
                self.y=self.y-1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==5:
            if self.map[self.x-1][self.y+1]!=1:
                if self.map[self.x-1][self.y+1]==4:
                    self.score=-50
                elif self.map[self.x-1][self.y+1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x-1
                self.y=self.y+1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==6:
            if self.map[self.x+1][self.y-1]!=1:
                if self.map[self.x+1][self.y-1]==4:
                    self.score=-50
                elif self.map[self.x+1][self.y-1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x+1
                self.y=self.y-1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        if command==7:
            if self.map[self.x+1][self.y+1]!=1:
                if self.map[self.x+1][self.y+1]==4:
                    self.score=-50
                elif self.map[self.x+1][self.y+1]==5:
                    self.score=-5
                else:
                    self.score=-1
                self.map[self.x][self.y]=5
                self.x=self.x+1
                self.y=self.y+1
                #self.score=-1
                if self.map[self.x][self.y]==2:
                    self.isDone=True
                    self.score=50
                self.map[self.x][self.y]=self.symbol
        
        