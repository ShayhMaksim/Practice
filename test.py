import pandas as pd

q_learning = pd.read_csv('Q_learning3')
sarsa =pd.read_csv('Sarsa')

shortQ=[]
shortS=[]
shortQ2=[]
shortS2=[]
iQ=[]
iS=[]
shortQ.append(q_learning['Полное вознаграждение'][0])
shortS.append(sarsa['Полное вознаграждение'][0])
shortQ2.append(q_learning['Длина эпизода'][0])
shortS2.append(sarsa['Длина эпизода'][0])
iQ.append(0)
iS.append(0)
for i in range(len(q_learning['Полное вознаграждение'])):
    if i%20==0:
        shortQ.append(q_learning['Полное вознаграждение'][i])
        shortS.append(sarsa['Полное вознаграждение'][i])
        shortQ2.append(q_learning['Длина эпизода'][i])
        shortS2.append(sarsa['Длина эпизода'][i])
        iQ.append(i)
        iS.append(i)

import matplotlib.pyplot as plt
plt.plot(iQ,shortQ,'r',label="Q-обучение")  # построение графика
plt.plot(iS, shortS,'b',label="SARSA")  # построение графика
plt.legend(loc=0)
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.savefig('Зависимсоть вознаграждения в эпизоде от времени x')
plt.show()

import matplotlib.pyplot as plt
plt.plot(iQ,shortQ2,'r',label="Q-обучение")  # построение графика
plt.plot(iS, shortS2,'b',label="SARSA")  # построение графика
plt.legend(loc=0)
plt.title('Зависимсоть длины эпизода от времени')
plt.xlabel('Эпизод')
plt.ylabel('Длина')
plt.savefig('Зависимсоть длины эпизода от времени x')
plt.show()