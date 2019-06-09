'''
Contains methods to plot results

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 8, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import numpy as np
from random import randint
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation

from utils.logwriter import Logger, MessageType


class Plotter:
    def __init__(self, console=False):
        '''
        Initializes logger and string for results path
        
        @param console: if True, prints debug in console
        '''
        self.logger = Logger(console=console)
        self.results_path = os.path.join(os.path.join(os.path.join(os.getcwd().split('SemanticEncoder')[0], 
                                                           'SemanticEncoder'), '_files'), 'results')
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        
    def plot_loss(self, history, tag=''):
        '''
        Plots loss and validation loss along training
        
        @param tag: string with folder name to saver results under
        @param history: dictionary with training history
        '''
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 10})
            
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Autoencoder Loss')
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend(['train', 'test'], loc='upper right')
            
            if tag and isinstance(tag, str):
                root = os.path.join(self.results_path, tag)
                file_name = os.path.join(root, 'ae_loss.png')
                if not os.path.isdir(root):
                    os.mkdir(root)
            else:
                file_name = os.path.join(self.results_path, 'ae_loss.png')
            
            plt.savefig(file_name)
        except OSError:
            self.logger.write_message('Loss image could not be saved under %s.' % file_name, MessageType.ERR)
    
    def plot_encoding(self, input_set, encoding, output_set, tag=None):
        '''
        Plots input example vs encoded example vs decoded example of 5 random examples
        in test set
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param output_set: autoencoder output
        @param tag: string with folder name to saver results under
        '''
        ex_idx = set()
        while len(ex_idx) < 5:
            ex_idx.add(randint(0, input_set.shape[0] - 1))
        
        error = output_set - input_set
    
        try:    
            plt.figure()
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(wspace=0.4, hspace=0.9)
            
            for i, idx in enumerate(ex_idx):
                ax = plt.subplot(5, 4, 4 * i + 1)
                plt.plot(input_set[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Input' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 2)
                plt.plot(encoding[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Encoding' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 3)
                plt.plot(output_set[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Output' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 4)
                plt.plot(error[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('Error')
                ax.axes.get_xaxis().set_visible(False)
            
            if tag and isinstance(tag, str):
                root = os.path.join(self.results_path, tag)
                file_name = os.path.join(root, 'ae_encoding.png')
                if not os.path.isdir(root):
                    os.mkdir(root)
            else:
                file_name = os.path.join(self.results_path, 'ae_encoding.png')
            
            plt.savefig(file_name)
        except OSError:
            self.logger.write_message('Error image could not be saved under %s.' % file_name, MessageType.ERR)
            
    def plot_spatial_distribution(self, input_set, encoding, output_set, labels, tag=None):
        '''
        Plots the spatial distribution of input, encoding and output using PCA and TSNE
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param output_set: autoencoder output
        @param labels: data set labels
        @param tag: string with folder name to saver results under
        ''' 
        plt.figure()
        plt.rcParams.update({'font.size': 8})
             
        try:    
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            pca = PCA(n_components=2)
             
            ax = plt.subplot(331)
            ax.set_title('PCA - Input')
            input_fts = pca.fit_transform(input_set)
            plt.scatter(input_fts[:,0], input_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
            
            ax = plt.subplot(332)
            ax.set_title('PCA - Encoding')
            encoding_fts = pca.fit_transform(encoding)
            plt.scatter(encoding_fts[:,0], encoding_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
             
            ax = plt.subplot(333)
            ax.set_title('PCA - Output')
            output_fts = pca.fit_transform(output_set)
            plt.scatter(output_fts[:,0], output_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
        except ValueError:
            self.logger.write_message('PCA could not be computed.', MessageType.ERR)
            
        try:
            tsne = TSNE(n_components=2)
             
            ax = plt.subplot(334)
            ax.set_title('TSNE - Input')
            input_fts = tsne.fit_transform(input_set)
            plt.scatter(input_fts[:,0], input_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))

            ax = plt.subplot(335)
            ax.set_title('TSNE - Encoding')
            encoding_fts = tsne.fit_transform(encoding)
            plt.scatter(encoding_fts[:,0], encoding_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
                         
            ax = plt.subplot(336)
            ax.set_title('TSNE - Output')
            output_fts = tsne.fit_transform(output_set)
            plt.scatter(output_fts[:,0], output_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))  
        except ValueError:
            self.logger.write_message('TSNE could not be computed.', MessageType.ERR)
        
        try:
            lda = LatentDirichletAllocation(n_components=2)

            ax = plt.subplot(337)
            ax.set_title('LDA - Input')
            input_fts = lda.fit_transform(input_set)
            plt.scatter(input_fts[:,0], input_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))

            ax = plt.subplot(338)
            ax.set_title('LDA - Encoding')
            encoding_fts = lda.fit_transform(encoding)
            plt.scatter(encoding_fts[:,0], encoding_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))

            ax = plt.subplot(339)
            ax.set_title('LDA - Output')
            output_fts = lda.fit_transform(output_set)
            plt.scatter(output_fts[:,0], output_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
        except ValueError:
            self.logger.write_message('LDA could not be computed.', MessageType.ERR)
            
        try:
            if tag and isinstance(tag, str):
                root = os.path.join(self.results_path, tag)
                file_name = os.path.join(root, 'ae_distribution.png')
                if not os.path.isdir(root):
                    os.mkdir(root)
            else:
                file_name = os.path.join(self.results_path, 'ae_distribution.png')
                
            if not os.path.isdir(self.results_path):
                os.mkdir(self.results_path)
                
            plt.savefig(file_name)
        except OSError:
            self.logger.write_message('Scatter plots could not be saved under %s.' 
                                      % file_name, MessageType.ERR)
            
    def plot_pca_vs_encoding(self, input_set, encoding, tag=None):
        '''
        Plots PCA against encoding components
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param tag: string with folder name to saver results under
        ''' 
        ex_idx = set()
        while len(ex_idx) < 5:
            ex_idx.add(randint(0, input_set.shape[0] - 1))
            
        pca = PCA(n_components=encoding.shape[1])
        input_fts = pca.fit_transform(input_set)
    
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(wspace=0.4, hspace=0.9)
            
            for i, idx in enumerate(ex_idx):
                ax = plt.subplot(5, 2, 2 * i + 1)
                plt.plot(input_fts[idx, :], linestyle='None', marker='o', markersize=3)
                ax.set_title('%d - PCA' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 2, 2 * i + 2)
                plt.plot(encoding[idx, :], linestyle='None', marker='o', markersize=3)
                ax.set_title('%d - Encoding' % idx)
                ax.axes.get_xaxis().set_visible(False)
            
            if tag and isinstance(tag, str):
                root = os.path.join(self.results_path, tag)
                file_name = os.path.join(root, 'ae_components.png')
                if not os.path.isdir(root):
                    os.mkdir(root)
            else:
                file_name = os.path.join(self.results_path, 'ae_components.png')
                
            plt.savefig(file_name)
        except (OSError, ValueError):
            self.logger.write_message('PCA vs Encoding image could not be saved under %s.' 
                                      % file_name, MessageType.ERR)
