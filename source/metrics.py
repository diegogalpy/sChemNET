import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '09-09-2021'


def myrecall(scores, y_true):
  #print('Total number of small molecules', len(y_true))
  topK = [10,50,100,200,300,400,500,600,700,800,900,1000]
  #idx_true = np.where(y_true > 0)
  index = np.flip(np.argsort(scores))
  recall = defaultdict(list)
  M = np.sum(y_true)
  for i in topK:
    # number of positives found in the topK
    tp = np.sum(y_true[index[0:i]])
    recall['x'].append(i)
    recall['y'].append(tp / float(M))
  
  #mean_recall = np.mean(recall['y'][0:2]) # mean in top 100
  return recall


def myrecall_topK(scores, y_true, topK, RNAtarget, DrugCID, methodname):
  #print('Total number of small molecules', len(y_true))
  
  #idx_true = np.where(y_true > 0)
  index = np.flip(np.argsort(scores))
  recall = pd.DataFrame()
  M = np.sum(y_true)
  
  for i in topK:
    # number of positives found in the topK
    tp = np.sum(y_true[index[0:i]])
    recall = recall.append({'topN': i, 'recall': tp / float(M), 'RNAtarget': RNAtarget, 'CID_test': DrugCID, 'method': methodname},ignore_index=True)
   
  
  #mean_recall = np.mean(recall['y'][0:2]) # mean in top 100
  return recall

def myauc(y_true, y_pred):

	y_true_tmp = y_true.flatten()
	scores = y_pred.flatten()

	y_true_tmp[y_true_tmp > 0] = 1

	fpr, tpr,_ = roc_curve(y_true_tmp, scores, pos_label = 1)
  
	df = pd.DataFrame()
	df["TPR"] = tpr
	df["FPR"] = fpr  

	return roc_auc_score(y_true_tmp, scores), df