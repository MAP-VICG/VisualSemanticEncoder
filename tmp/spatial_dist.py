'''
Created on Jun 18, 2019

@author: damaresresende
'''
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Concatenate, BatchNormalization

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from core.vsencoder import SemanticEncoder

def create_model(data):
    input_vis_fts = Input(shape=(data[:,:2048].shape[1],))
    input_sem_fts = Input(shape=(data[:,2048:].shape[1],1))
    embedding_sem = Conv1D(5, 25, use_bias=False, padding='same')(input_sem_fts)
    
    flatten_sem = Flatten()(embedding_sem)
    encoded = Concatenate()([input_vis_fts, flatten_sem])
    
    encoded = Dense(1426, activation='relu')(encoded)
    encoded = Dense(732, activation='relu')(encoded)
    encoded = Dense(328, activation='relu')(encoded)
    
    encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
    
    encoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=encoded)
    return encoder

def load_data():
    data = []
    with open('../_files/results/test_set.txt') as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data.append([float(v) for v in line.split(',')])
               
    with open('../_files/results/test_labels.txt') as f:
        labels = [float(v) for v in f.readline().split(',')]
    
    return np.array(data), np.array(labels)
        
def filter_data(data, labels):
    # horse, blue+whale, giraffe, zebra, dolphin
    classes = [7, 9, 31, 38, 50]
    mask = [False] * labels.shape[0]
    
    for idx, lb in enumerate(labels):
        if lb in classes:
            mask[idx] = True
    
    return data[mask, :], labels[mask]

def plot_spatial_distribution(encoding, labels):
    classes = [7, 9, 31, 38, 50]
    colors = ['#606060', '#0000CC', '#CC0000', '#00CC00', '#FF8000']
    
    try:
        plt.figure()
        pca = PCA(n_components=2)
         
        encoding_fts = pca.fit_transform(encoding)

                
        for i, c in enumerate(classes):
            mask = []
            for lb in labels:
                if lb == c:
                    mask.append(True)
                else:
                    mask.append(False)
            
            plt.scatter(encoding_fts[mask,0], encoding_fts[mask,1], c=colors[i], s=np.ones(labels.shape) * 25)
        
        plt.legend(['horse', 'blue+whale', 'giraffe', 'zebra', 'dolphin'], loc='top left', fontsize=7)
        plt.show()      
    except ValueError as e:
        print(e)
         
    try:
        plt.figure()
        tsne = TSNE(n_components=2)
          
        encoding_fts = tsne.fit_transform(encoding)
        
        for i, c in enumerate(classes):
            mask = []
            for lb in labels:
                if lb == c:
                    mask.append(True)
                else:
                    mask.append(False)
            
            plt.scatter(encoding_fts[mask,0], encoding_fts[mask,1], c=colors[i], s=np.ones(labels.shape) * 25)
            
        plt.legend(['horse', 'blue+whale', 'giraffe', 'zebra', 'dolphin'], loc='top left', fontsize=7)
        plt.show()                 
    except ValueError as e:
        print(e)

data, lbs = load_data()
data, lbs = filter_data(data, lbs)

enc = SemanticEncoder(50, 128)
data = enc.pick_semantic_features('COLOR', data, opposite=True)

model = create_model(data)
model.load_weights('../_files/results/2I/_COLOR/encoder.h5')
enc = model.predict([data[:,:2048], np.expand_dims(data[:,2048:], axis=-1)])

plot_spatial_distribution(enc, lbs)
    
