import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from collections import defaultdict 
import NNmodels as md
import metrics as me
from tqdm import tqdm
import warnings
import dataProcessing as dp


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '05-04-2021'

def get_rank(score, label):

  index = np.flip(np.argsort(score))

  idx_test = np.where(label > 0)[0][0]



  position_rank = np.where(index == idx_test)[0][0]

  return position_rank



def get_y_seqSIM(y_train, seqSIM, i):

  # prepare sequence similarity matrix

  y_s = y_train.copy()



  for c in range(y_s.shape[1]):

    if c != i:

      y_s[:,c] = np.multiply(seqSIM[i,c], y_s[:,c])

  return y_s.astype(np.float32)



def prepare_split(idx_ones, idx_zeros, Y, X, j):

  idx_test = np.concatenate(([idx_ones[j]], idx_zeros[0:Nneg_test]), axis = 0)

  idx_ones_left = np.delete(idx_ones, np.where(idx_ones == idx_ones[j]))



  idx_train = np.concatenate((idx_ones_left, idx_zeros[Nneg_test:Nneg_test+2000]), axis = 0)



  # random permutation

  idx_test  = np.random.permutation(idx_test)

  idx_train  = np.random.permutation(idx_train)



  y_train = Y[idx_train, :]

  y_test = Y[idx_test, :]



  x_train = X[idx_train,:]

  x_test = X[idx_test,:]



  return y_train, y_test, x_train, x_test

def sChemNET(y_train, x_train, y_test, x_test, learning_rate, n_epochs, n_units, prob_dropout, alpha, i, SeqSIM  ):
  model = md.MF2MLP( y_train, x_train)
  model.loss = 'custom_seqSIM'
  model.Y_s = get_y_seqSIM(y_train, SeqSIM.values, i)
  model.optimiser = 'ADAM'
  model.learning_rate = learning_rate
  model.EPOCHS = n_epochs
  model.input_dim = x_train.shape[1]
  model.hidden_layer_sizes = [n_units, y_train.shape[1]]
  model.prob_drop = prob_dropout
  model.alpha = alpha


  # return the trained model
  trained_model =  model.train_MLP()

   # predictions on training set
  Yhat_train = trained_model.predict(x_train)
  score_train = Yhat_train[:, i]
  labels_train = y_train[:, i]

  # predictions on test set
  Yhat_test = trained_model.predict(x_test)

  score_test = Yhat_test[:,i]
  label_test = y_test[:,i]

  return labels_train, label_test,score_train,score_test, model

def generate_predictions(X, Y, column_names, row_names, i, best_modelZseqSIM, SeqSIM):
  # number of samples for predictions
  Nneg = 4000
  idx_ones = np.where(Y[:, i] > 0)[0]
  idx_zeros = np.where(Y[:, i] == 0)[0]

  for j in tqdm(range(20)):
    # random permutation
    idx_ones  = np.random.permutation(idx_ones)
    idx_zeros  = np.random.permutation(idx_zeros)

    idx_test = idx_zeros[0:Nneg]
    idx_train = np.concatenate((idx_ones, idx_zeros[Nneg:Nneg+2000]), axis = 0)

    # random permutation
    idx_test  = np.random.permutation(idx_test)
    idx_train  = np.random.permutation(idx_train)

    row_names_permuted = [row_names[r] for r in idx_test]
    y_train = Y[idx_train, :]
    y_test = Y[idx_test, :]

    x_train = X[idx_train,:]
    x_test = X[idx_test,:]

    (labels_train, label_test,score_train,score_test, model) = sChemNET(y_train.copy(), x_train.copy(),
                            y_test.copy(), x_test.copy(),
                            best_modelZseqSIM['params']['lr'],
                            int(best_modelZseqSIM['params']['EPOCHS']),
                            16, best_modelZseqSIM['params']['drop'],
                            best_modelZseqSIM['params']['alph'], i, SeqSIM )

    scores =  np.squeeze(score_test)
    result = defaultdict(list)
    for u in range(len(scores)):
      result['miRNA'].append(column_names[i])
      result['CID'].append(row_names_permuted[u])
      result['score'].append(scores[u])

    if j == 0:
      df_result = pd.DataFrame(result)
    else:
      df_result = pd.concat([df_result, pd.DataFrame(result)],join="inner", ignore_index=True)

  df_result = df_result.groupby(['miRNA', 'CID'],as_index=False,sort=False)['score'].mean()
  df_result['percentile score'] = scores2percentile(df_result.copy())
  return df_result

def scores2percentile(df_result):
  # calculate percentile of scores
  from scipy import stats
  percentiles = list()
  allscores = list(df_result['score'])
  for i in range(len(allscores)):
    percentiles.append(stats.percentileofscore(allscores, allscores[i]))

  return percentiles

def combine_predictions_withdrugRepoHub_dataset(root_dir, df_result):

  df_repo_hub = pd.read_csv(root_dir + '/data/repurposing_samples_20200324.txt', sep="\t", encoding="ISO-8859-1")
  df_repo_hub_drugs = pd.read_csv(root_dir + '/data/repurposing_drugs_20200324.txt', sep="\t", encoding="ISO-8859-1")
  df_repo_hub = df_repo_hub.rename({'pubchem_cid': 'CID'}, axis='columns')

  df_result2 = pd.merge(df_result, df_repo_hub[['pert_iname', 'CID']], on='CID', how='left').fillna(0)
  df_result2 = df_result2.drop_duplicates(subset=['miRNA', 'CID', 'score'], ignore_index=True)
  df_result3 = pd.merge(df_result2, df_repo_hub_drugs, on='pert_iname')
  df_result3 = df_result3.sort_values(by=['miRNA', 'score'], ascending=False, ignore_index=True)
  df_result3 = df_result3.rename(columns={'score': 'sChemNET score'})

  

  return df_result3