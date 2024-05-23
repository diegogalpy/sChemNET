import pandas as pd 
import pathlib

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from keras.metrics import categorical_accuracy
from keras.layers import BatchNormalization

import scipy


import MFmodel as MF
#import metrics 
from collections import defaultdict

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '05-10-2021'

class MF2MLP(object):
    
    def __init__(self, R_train, X_train):
        self.R_train = R_train    
        self.X_train = X_train
        self.k = None
        self.alpha = None
        self.hidden_layer_sizes = None     
        self.l2 = None
        self.optimiser = None
        self.EPOCHS = None
        self.learning_rate = None
        self.model = None
        self.W = None
        self.H = None
        self.history = None
        self.loss = 'mse'
        self.input_dim = None
        self.prob_drop = None
        
        return
    
    def train_MF(self):
        # Step 1: train the MF model
        print('Training the MF model...')
        [self.W, self.H, J] = MF.DecompositionAlgorithm(self.R_train, self.k, self.alpha);
        return self.W, self.H, J
    
    def train_MLP(self):
         
        
        # Step 2: train the NN model
        #print('Training the MLP on the drug signatures')
        model = self.build_model()
        #print(model.summary())
        if self.optimiser == 'ADAM':
          optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif self.optimiser == 'SGD':
          optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        else:
          optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
        
        if self.loss == 'mse':
          model.compile(loss= 'mse',
                  optimizer=optimizer,
                  metrics=['mse'])
          self.history = model.fit(self.X_train, self.R_train, batch_size = self.X_train.shape[0],
                              epochs=self.EPOCHS, verbose = 0, shuffle = False)
        
        elif self.loss == 'cross_entropy':
          model.compile(loss= 'categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=categorical_accuracy)
          self.history = model.fit(self.X_train, self.R_train, batch_size = self.X_train.shape[0],
                              epochs=self.EPOCHS, verbose = 0, shuffle = False)
        elif self.loss == 'custom':
            model.compile(loss= self.my_loss_fn,
                    optimizer=optimizer,
                    metrics=['mse'])
            self.history = model.fit(self.X_train, self.R_train, batch_size = self.X_train.shape[0], 
                                epochs=self.EPOCHS, verbose = 0, shuffle = True)
    
          
        
       
        
        
        return model
        
    def build_model(self):
   
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(self.hidden_layer_sizes[0], input_dim=self.input_dim, activation = 'relu'))
        model.add(tf.keras.layers.Dropout(self.prob_drop))   
        model.add(BatchNormalization())

        for i, layer_size in enumerate(self.hidden_layer_sizes[1:]):
            if i == len(self.hidden_layer_sizes[1:])-1:
              model.add(tf.keras.layers.Dense(layer_size, activation='sigmoid'))
              model.add(tf.keras.layers.Dropout(self.prob_drop)) 
              
            else:
              model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
              model.add(tf.keras.layers.Dropout(self.prob_drop))
              model.add(BatchNormalization()) 
            
                  
        return model
    
    def plot_cost_function(self, mytitle = ''):
        plt.figure(figsize=(5,5))
        plt.plot(self.history.history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title(mytitle)
        plt.grid(True)
        plt.show()
    '''  
    def compare_signatures(self, trained_MLP, W):
        W_predicted = trained_MLP.predict(self.X_train)
        x = W_predicted.flatten()
        y = W.flatten()
        plt.figure(figsize=(5,5))
        plt.scatter(W_predicted.flatten(), W.flatten())
        plt.xlabel('predicted drug signature (MLP)')
        plt.ylabel('Target drug signature (MF)')
        plt.title('Training set')
        plt.grid(True)
        plt.show()

        print('Pearson Correlation', scipy.stats.pearsonr(x,y))
   '''

    def my_loss_fn(self, y_true, y_pred):
        #print(y_true.shape, y_pred.shape)
        mask1 = tf.cast(y_true > 0.0, y_true.dtype)
        mask2 = tf.cast(y_true == 0.0, y_true.dtype)
        
        y_pred_masked1 = tf.math.multiply(y_pred, mask1)
        y_pred_masked2 = tf.math.multiply(y_pred, mask2)
        
        squared_difference = tf.square(y_true - y_pred_masked1) +  tf.math.multiply(self.alpha, tf.square(y_pred_masked2))
        loss = tf.math.multiply(1/(2*self.R_train.shape[0]), tf.reduce_sum(squared_difference))  # Note the `axis=-1`

        #tf.Print(loss, [loss], "Inside loss function")
        return loss