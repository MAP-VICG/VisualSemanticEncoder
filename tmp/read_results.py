'''
Created on Jul 27, 2019

@author: damaresresende
'''
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from core.vsencoder import SemanticEncoder


def create_model(data):
    input_fts = Input(shape=(data.shape[1],))
    
    encoded = Dense(1426, activation='relu')(input_fts)
    encoded = Dense(732, activation='relu')(encoded)
    encoded = Dense(328, activation='relu')(encoded)
    
    encoded = Dense(128, activation='relu')(encoded)
    
    encoder = Model(inputs=input_fts, outputs=encoded)
    return encoder

def load_data():
    data = []
    with open('../_files/results/_datasets/test_set.txt') as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data.append([float(v) for v in line.split(',')])
               
    with open('../_files/results/_datasets/test_labels.txt') as f:
        labels = [float(v) for v in f.readline().split(',')]
    
    return np.array(data), np.array(labels)

data, lbs = load_data()

enc = SemanticEncoder(50, 128)
data = enc.pick_semantic_features('ALL', data, opposite=False)

dirc = '/Users/damaresresende/Desktop/normalizacao/con_L2_C/ALL/'

model = create_model(data)
model.load_weights(dirc + 'encoder.h5')

code = model.apply(data)
code_mean = np.mean(code, axis=0)
code_std = np.std(code, axis=0)
code_cov = np.cov(np.array(code).transpose())

upperlimits = [True, False] * 64
lowerlimits = [False, True] * 64

lbs = ['mean', 'std']

plt.figure(figsize=(14, 12))
plt.matshow(code_cov, fignum=1)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=12)
plt.savefig(dirc + '_correlation_mat.png', bbox_inches='tight')


plt.figure(figsize=(14, 12))
plt.errorbar([x for x in range(128)], code[0], code[1], fmt='o')

plt.legend(loc='top right')
plt.xlabel('Encoding Dimension', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.savefig(dirc + '_mean_std.png', bbox_inches='tight')
# plt.show()

# plt.figure()
# plt.scatter([x for x in range(128)], code_mean)
# plt.scatter([x for x in range(128)], code_std)
# 
# plt.legend(lbs, loc='top right')
# plt.xlabel('Encoding Dimension', fontsize=12)
# plt.ylabel('Amplitude', fontsize=12)
# plt.savefig(dirc + 'mean_std_two.png', bbox_inches='tight')
# # plt.show()


print('done')