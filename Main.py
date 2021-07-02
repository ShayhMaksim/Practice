from Map import Map
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QBrush, QColor
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget,QApplication,QGraphicsView,QGraphicsScene
import sys
from BadZone import BadZone
mapp=Map.reset()
# zone1=BadZone(30,30,40,40)
# zone2=BadZone(15,15,18,30)
# zone3=BadZone(18,30,30,32)
# mapp.AddWalls([zone1,zone2,zone3])

app = QApplication(sys.argv)
scene = QGraphicsScene()
graphicsView = QGraphicsView(scene)
graphicsView.show()
graphicsView.resize(1000,1000)

i=0
j=0
c=18

while i<50:
    while j<50:
        if mapp.chart[i][j]==1:
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,0,255),QBrush(QColor(0,0,255)))
        if mapp.chart[i][j]==2:
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,0,0),QBrush(QColor(255,0,0)))
        if mapp.chart[i][j]==3:
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,0),QBrush(QColor(0,255,0)))
        if mapp.chart[i][j]==4:
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,155,150),QBrush(QColor(255,155,150)))
        if mapp.chart[i][j]==0:
            scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,255))
        j=j+1
    i=i+1
    j=0

# a=mapp.step(2)
# b=mapp.decode(a[0])

#mapp.step(3)
#mapp.step(4)

def Update():
    scene.clear()
    i=0
    j=0
    while i<50:
        while j<50:
            if mapp.chart[i][j]==1:
                scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,0,255),QBrush(QColor(0,0,255)))
            if mapp.chart[i][j]==2:
                scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,0,0),QBrush(QColor(255,0,0)))
            if mapp.chart[i][j]==3:
                scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,0),QBrush(QColor(0,255,0)))
            if mapp.chart[i][j]==4:
                scene.addRect(QRectF(i*c,j*c,c,c),QColor(255,155,150),QBrush(QColor(255,155,150)))
            if mapp.chart[i][j]==0:
                scene.addRect(QRectF(i*c,j*c,c,c),QColor(0,255,255))
            j=j+1
        i=i+1
        j=0

Update()
#graphicsView.update()



sys.exit(app.exec())