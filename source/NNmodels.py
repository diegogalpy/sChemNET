import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

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

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '08-12-2021'


class MF2MLP(object):
    
    def __init__(self, R_train, X_train):
        self.R_train = R_train.astype(np.float64)    
        self.X_train = X_train.astype(np.float64)
        self.Y_s = None
        self.k = None
        self.alpha = None
        self.hidden_layer_sizes = None     
        self.l2 = None
        self.optimiser = None
        self.EPOCHS = None
        self.learning_rate = None
        self.model = None
        self.batch_size = None
        self.W = None
        self.H = None
        self.history = None
        self.loss = 'mse'
        self.input_dim = None
        self.prob_drop = None
        
        return
    
        
    def train_MLP(self):
         
        
        # Step 2: train the NN model
        #print('Training the MLP on the drug signatures')
        #print(self.X_train[0,:])
        #print('training sets...', self.R_train.shape, self.X_train.shape)
        model = self.build_model()
        #print(model.summary())
        if self.optimiser == 'ADAM':
          optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif self.optimiser == 'SGD':
          optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        elif self.optimiser == 'Ftrl':
          optimizer = tf.keras.optimizers.Ftrl(self.learning_rate)
        else:
          optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
        
        #print(self.loss)
        if self.loss == 'mse':
          model.compile(loss= 'mse',
                  optimizer=optimizer,
                  metrics=['mse'])
          self.history = model.fit(self.X_train, self.R_train, batch_size = self.batch_size,
                              epochs=self.EPOCHS, verbose = 0, shuffle = True)

        
        elif self.loss == 'cross_entropy':
          model.compile(loss= 'categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=categorical_accuracy)
          self.history = model.fit(self.X_train, self.R_train, batch_size = self.batch_size,
                              epochs=self.EPOCHS, verbose = 0, shuffle = True)
                              
        elif self.loss == 'custom':
            model.compile(loss= self.my_loss_fn,
                    optimizer=optimizer,
                    metrics=['mse'])
            self.history = model.fit(self.X_train, self.R_train, batch_size = self.X_train.shape[0],
                                epochs=self.EPOCHS, verbose = 0, shuffle = True)
        elif self.loss == 'custom_seqSIM':
            model.compile(loss= self.my_loss_fn_seq_sim,
                    optimizer=optimizer,
                    metrics=['mse'])
            # VERY important: DO NOT SHUFFLE training otherwise will not work
            self.history = model.fit(self.X_train, self.R_train, batch_size = self.X_train.shape[0],
                                epochs=self.EPOCHS, verbose = 0, shuffle = False)
        else:
            print('error')  
        
        return model
        
    def build_model(self):        
        model = tf.keras.models.Sequential()        
        model.add(tf.keras.layers.Dense(self.hidden_layer_sizes[0], input_dim = self.input_dim, activation = 'relu'))
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
    
    def my_loss_fn(self, y_true, y_pred):
      
      y_true = tf.cast(y_true, tf.float64)
      y_pred = tf.cast(y_pred, tf.float64)
      
      mask1 = tf.cast(y_true > 0, y_true.dtype)
      mask2 = tf.cast(y_true == 0, y_true.dtype)
      
      y_pred_masked1 = tf.math.multiply(y_pred, mask1)
      y_pred_masked2 = tf.math.multiply(y_pred, mask2)
      
      y_pred_masked1 = tf.cast(y_pred_masked1, tf.float64)
      y_pred_masked2 = tf.cast(y_pred_masked2, tf.float64)
      alpha_tf = tf.cast(self.alpha, tf.float64)

      squared_difference = tf.square(y_true - y_pred_masked1) +  tf.math.multiply(alpha_tf, tf.square(y_pred_masked2))
      
      constant = tf.cast(1/float(2*y_true.shape[0]), tf.float64)
      squared_difference = tf.cast(squared_difference, tf.float64)

      loss = tf.math.multiply(constant, tf.reduce_sum(squared_difference)) 
      
      
      return loss 

    def my_loss_fn_seq_sim(self, y_true, y_pred):
        #print(y_true.shape)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mask1 = tf.cast(y_true > 0, y_true.dtype)
        mask2 = tf.cast(y_true == 0, y_true.dtype)
        
        y_pred_masked1 = tf.math.multiply(y_pred, mask1)
        y_pred_masked2 = tf.math.multiply(y_pred, mask2)
        
        y_pred_masked1 = tf.cast(y_pred_masked1, tf.float32)
        y_pred_masked2 = tf.cast(y_pred_masked2, tf.float32)

        alpha_tf = tf.cast(self.alpha, tf.float32)

        squared_difference = tf.square(tf.math.multiply(self.Y_s, (y_true - y_pred_masked1))) +  tf.math.multiply(alpha_tf, tf.square(y_pred_masked2))
        
        constant = tf.cast(1/float(2*y_true.shape[0]), tf.float32)
        squared_difference = tf.cast(squared_difference, tf.float32)

        loss = tf.math.multiply(constant, tf.reduce_sum(squared_difference)) 
        
        
        return loss