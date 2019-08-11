'''
Created on Jul 29, 2019

@author: damaresresende
'''
import numpy as np
import matplotlib.pyplot as plt


cats = ['ALL', 'COLOR', 'TEXTURE', 'PARTS', 'SHAPE', '_COLOR', '_TEXTURE', '_PARTS', '_SHAPE']
loss = {key: np.zeros((50,10)) for key in cats}
acc = {key: np.zeros((50,10)) for key in cats}

res_loss = {key: [] for key in cats}
res_acc = {key: [] for key in cats}

for i in range(10):
    for flag in cats:
        with open('/Users/damaresresende/Desktop/HISTORY/f' + str(i) + '/' + flag + '/history.txt') as f:
            for row in f.readlines():
                if row.startswith('loss'):
                    for k, v in enumerate((row.split('loss:')[1]).split(',')):
                        loss[flag][k,i] = float(v)
#                     loss[flag].append([float(v) for v in (row.split('loss:')[1]).split(',')])
                if row.startswith('acc'):
                    for k, v in enumerate((row.split('acc:')[1]).split(',')):
                        acc[flag][k,i] = float(v)
#                     acc[flag].append([float(v) for v in (row.split('acc:')[1]).split(',')])

for flag in cats:
    res_loss[flag].append(np.average(loss[flag], 1))
    res_loss[flag].append(np.std(loss[flag], 1))
    
    res_acc[flag].append(np.average(acc[flag], 1))
    res_acc[flag].append(np.std(acc[flag], 1))
    
upperlimits = [True, False] * 25
lowerlimits = [False, True] * 25

lbs = ['Cor', 'Textura', 'Partes', 'Forma']

for i, flag in enumerate(['COLOR', 'TEXTURE', 'PARTS', 'SHAPE']):
    plt.errorbar([x for x in range(50)], res_loss[flag][0], res_loss[flag][1], uplims=upperlimits, lolims=lowerlimits, label=lbs[i])

plt.legend(loc='top right')
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
# plt.ylabel('Acurácia', fontsize=14)
plt.show()

lbs = ['Sem Cor', 'Sem Textura', 'Sem Partes', 'Sem Forma']

for i, flag in enumerate(['_COLOR', '_TEXTURE', '_PARTS', '_SHAPE']):
    plt.errorbar([x for x in range(50)], res_loss[flag][0], res_loss[flag][1], uplims=upperlimits, lolims=lowerlimits, label=lbs[i])

plt.legend(loc='top right')
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
# plt.ylabel('Acurácia', fontsize=14)
plt.show()
             
print('done')