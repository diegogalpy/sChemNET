import pandas as pd 
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix


def get_splits_columns_seqSIM(X, Y, seqSIM):
  

  Y_train = defaultdict()
  Y_dev = defaultdict()
  Y_test = defaultdict()
  X_train = defaultdict()
  X_dev =defaultdict()
  X_test = defaultdict()
  S_train = defaultdict()
  
  for i in range(Y.shape[1]):    
        
    idx_ones = np.where(Y[:, i] > 0)[0]
    idx_zeros = np.where(Y[:, i] == 0)[0]
          
    idx_ones  = np.random.permutation(idx_ones)
    idx_zeros  = np.random.permutation(idx_zeros)
          
    if len(idx_ones) == 5:
      n_train_ones = 3
      n_dev_ones = 1
    else:
      n_train_ones = int(np.fix(len(idx_ones)*0.8))
      n_dev_ones = int(np.fix((len(idx_ones)-n_train_ones)*0.5))
      if n_dev_ones == 0: n_dev_ones = 1
      
    #n_train_zeros = int(np.fix(len(idx_zeros)*0.8))
    #n_dev_zeros = int(np.fix((len(idx_zeros)-n_train_zeros)*0.5))
    n_train_zeros = len(idx_zeros) - 4000 
    n_dev_zeros = 2000 

    idx_train = np.concatenate((idx_ones[0:n_train_ones], idx_zeros[0:n_train_zeros]), axis = 0)
    idx_dev = np.concatenate((idx_ones[n_train_ones:n_train_ones + n_dev_ones], idx_zeros[n_train_zeros: n_train_zeros + n_dev_zeros]),  axis = 0)
    idx_test = np.concatenate((idx_ones[n_train_ones + n_dev_ones::], idx_zeros[n_train_zeros + n_dev_zeros::]), axis = 0)
    
    y_train = Y[idx_train, :]
    y_dev = Y[idx_dev, :]
    y_test = Y[idx_test, :]

    x_train = X[idx_train,:]
    x_dev = X[idx_dev,:]
    x_test = X[idx_test,:]     
    
    # prepare sequence similarity matrix
    y_s = y_train.copy()

    for c in range(y_s.shape[1]):
      if c != i:
        y_s[:,c] = np.multiply(seqSIM[i,c], y_s[:,c])
      
    # random permutation to avoid learning ordering    
    p = np.random.permutation(y_train.shape[0])
    y_train = y_train[p, :]
    y_s = y_s[p,:]
    x_train = x_train[p,:]

    p = np.random.permutation(y_dev.shape[0])
    y_dev = y_dev[p, :]
    x_dev = x_dev[p,:]

    p = np.random.permutation(y_test.shape[0])
    y_test = y_test[p, :]
    x_test = x_test[p,:]
    
    Y_train[i] = csr_matrix(y_train, dtype=np.int64)
    Y_dev[i] = csr_matrix(y_dev, dtype=np.int64)
    Y_test[i] = csr_matrix(y_test, dtype=np.int64)
    X_train[i] = csr_matrix(x_train, dtype=np.float64)
    X_dev[i] = csr_matrix(x_dev, dtype=np.float64)
    X_test[i] = csr_matrix(x_test, dtype=np.float64)
    S_train[i] = csr_matrix(y_s, dtype=np.float64)

  mysplits = defaultdict()
  mysplits['Y_train'] = Y_train
  mysplits['Y_dev'] = Y_dev
  mysplits['Y_test'] = Y_test
  mysplits['X_train'] = X_train
  mysplits['X_dev'] = X_dev
  mysplits['X_test'] = X_test
  mysplits['S_train'] = S_train
  return mysplits

def get_splits_columns(X, Y):
  

  Y_train = defaultdict()
  Y_dev = defaultdict()
  Y_test = defaultdict()
  X_train = defaultdict()
  X_dev =defaultdict()
  X_test = defaultdict()
  
  for i in range(Y.shape[1]):    
        
    idx_ones = np.where(Y[:, i] > 0)[0]
    idx_zeros = np.where(Y[:, i] == 0)[0]
          
    idx_ones  = np.random.permutation(idx_ones)
    idx_zeros  = np.random.permutation(idx_zeros)
          
    if len(idx_ones) == 5:
      n_train_ones = 3
      n_dev_ones = 1
    else:
      n_train_ones = int(np.fix(len(idx_ones)*0.8))
      n_dev_ones = int(np.fix((len(idx_ones)-n_train_ones)*0.5))
      if n_dev_ones == 0: n_dev_ones = 1
      
    #n_train_zeros = int(np.fix(len(idx_zeros)*0.8))
    #n_dev_zeros = int(np.fix((len(idx_zeros)-n_train_zeros)*0.5))
    n_train_zeros = len(idx_zeros) - 4000 
    n_dev_zeros = 2000 

    idx_train = np.concatenate((idx_ones[0:n_train_ones], idx_zeros[0:n_train_zeros]), axis = 0)
    idx_dev = np.concatenate((idx_ones[n_train_ones:n_train_ones + n_dev_ones], idx_zeros[n_train_zeros: n_train_zeros + n_dev_zeros]),  axis = 0)
    idx_test = np.concatenate((idx_ones[n_train_ones + n_dev_ones::], idx_zeros[n_train_zeros + n_dev_zeros::]), axis = 0)
    
    y_train = Y[idx_train, :]
    y_dev = Y[idx_dev, :]
    y_test = Y[idx_test, :]

    x_train = X[idx_train,:]
    x_dev = X[idx_dev,:]
    x_test = X[idx_test,:]     
    
    # random permutation to avoid learning ordering    
    p = np.random.permutation(y_train.shape[0])
    y_train = y_train[p, :]
    x_train = x_train[p,:]

    p = np.random.permutation(y_dev.shape[0])
    y_dev = y_dev[p, :]
    x_dev = x_dev[p,:]

    p = np.random.permutation(y_test.shape[0])
    y_test = y_test[p, :]
    x_test = x_test[p,:]
    
    Y_train[i] = csr_matrix(y_train, dtype=np.int64)
    Y_dev[i] = csr_matrix(y_dev, dtype=np.int64)
    Y_test[i] = csr_matrix(y_test, dtype=np.int64)
    X_train[i] = csr_matrix(x_train, dtype=np.float64)
    X_dev[i] = csr_matrix(x_dev, dtype=np.float64)
    X_test[i] = csr_matrix(x_test, dtype=np.float64)

  splits = defaultdict()
  splits['Y_train'] = Y_train
  splits['Y_dev'] = Y_dev
  splits['Y_test'] = Y_test
  splits['X_train'] = X_train
  splits['X_dev'] = X_dev
  splits['X_test'] = X_test

  return splits