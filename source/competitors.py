import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '20-08-2021'


def random_scores(N):
  '''
  Random scores sample from a uniform distribution
  N = number of elements
  '''
  return np.random.rand(N)

def chemical_similarity(X_train, y_train, X_test):
  '''

  This baseline calculates the Tanimoto similarity between the positive small 
  molecules in the training set and those in the testing set.
  Then, the score assigned to each small molecule in the testing set is the max 
  chemical similarity to any positive small molecule in training.

  score small molecule j = arg max i {SIM ij}, i represent a positive SM in training

  ''' 
  idx = np.where(y_train > 0)[0]
  tmp = X_train[idx,:].copy()
  #print(tmp.shape)
  x_t = np.vstack([X_test, tmp])

  n_test_samples = X_test.shape[0]

  chemSIM = 1-squareform(pdist(x_t, 'jaccard'))
  np.fill_diagonal(chemSIM, 0)

  # the part I care about
  chemSIM = chemSIM[0:n_test_samples, n_test_samples::]
  
  return chemSIM, np.amax(chemSIM, axis=1) 
  
def zhou_diffussion(X, y, alpha):

  W = 1-squareform(pdist(X, 'jaccard')) 
  #W = np.exp(-np.divide(np.power(d, 2),2))
  np.fill_diagonal(W, 0)

  D = np.diag(np.power(np.sum(W, axis = 0), -0.5)) # D^-0.5
 
  S = np.dot(np.dot(D, W), D)
  I = np.identity(S.shape[0])

  
  F = np.dot(np.linalg.inv(I - np.multiply(alpha, S)), y)
  return F