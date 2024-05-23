import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
from Bio import Align

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '20-08-2021'

def data_augmentation_non_replicated_repoHub(X_train, Y_train, X_repo, CID_SM2miR, CID_repo):

  X_train_aug = X_train
  Y_train_aug = Y_train.astype(float)
  CIDs = [i for i in CID_SM2miR]
  c = 0
  for j in range(X_repo.shape[0]):
    ban = 0
    for i in range(X_train.shape[0]):
      if (X_train[i,:] == X_repo[j,:]).all():
        ban = 1
        c+=1
        break

    if ban == 0:
      CIDs.append(CID_repo[j])
      X_train_aug = np.vstack([X_train_aug, X_repo[j,:]])
      Y_train_aug = np.vstack([Y_train_aug, np.zeros(shape = (1, Y_train.shape[1]))])

  print(X_train_aug.shape, Y_train_aug.shape, c)
  return X_train_aug, Y_train_aug, CIDs


def reduce_input_pca(Xt, pcs):
  pca = PCA(n_components=pcs)
  pca.fit(Xt)

  plt.bar(list(range(1,pcs+1)), pca.explained_variance_ratio_)
  plt.grid(True)
  plt.ylabel('Explained variance ratio')
  plt.xlabel('Principal components')
  plt.show()
  print(np.sum(pca.explained_variance_ratio_))
  X_all = pca.fit_transform(Xt)
  normed = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)

  return normed

def structural_similarity_profile(X, k):
  from scipy.spatial.distance import pdist, squareform
  
  ChemSIM = 1-squareform(pdist(X, 'jaccard')) # tanimoto similarity
  return reduce_input_pca(ChemSIM, k)

def get_sequence_similarities(miRNAs, root_dir):
  
  
  '''
  get the sequence similarities between all miRNAs
  '''
  aligner = Align.PairwiseAligner()
  aligner.mode = 'global'
  print('Algorithm =', aligner.algorithm)

  # READ THE SEQUENCES
  miRFasta = list()
  filename = root_dir + '/data/mature.fa'

  with open(filename) as f:
      content = f.readlines()
      miRFasta.append(content)  

  miRSeqs = defaultdict()
  ban = 0
  for i, content in enumerate(miRFasta[0]): 
      content = content.split(' ')
      if ban == 1:
          miRSeqs[seqID] = content[0].strip()
          ban = 0
      else:        
          ban = 1
          seqID = content[1] 

  # CALCULATE
  SeqSim = np.zeros((len(miRNAs), len(miRNAs)))

  for i,seq1 in enumerate(miRNAs):
      for j,seq2 in enumerate(miRNAs):
          try:       
            SeqSim[i,j] = aligner.score(miRSeqs[seq1], miRSeqs[seq2])
          except:
            continue

  # we normalise so that the seqSIM is between 0 and 1
  #SeqSim_normalised = np.divide(SeqSim, np.max(SeqSim, axis = 0))

  # we noormalised so that seqSIM is bettwn a and 1.
  a = 0.7
  xmin = np.min(SeqSim, axis = 0)
  xmax = np.max(SeqSim, axis = 0)

  m = np.divide((a-1), xmin - xmax)
  SeqSim_normalised = np.multiply(m, SeqSim) + (1-m*xmax)
  #np.fill_diagonal(SeqSim_normalised, 0)
  return pd.DataFrame(data=SeqSim_normalised,    # values
                      index=miRNAs,   
                      columns=miRNAs)  