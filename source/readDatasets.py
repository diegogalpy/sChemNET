import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from collections import defaultdict

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '09-09-2020'

def read_datasets(root): 
	df_SM2miR = pd.read_csv(root + '/data/SM2miR3_2015.txt', sep="\t", encoding="ISO-8859-1")
	# filter based on available CID and miRBase
	df_SM2miR = df_SM2miR[df_SM2miR[['CID', 'miRBase']].notnull().all(1)]
	# fix column names
	df_SM2miR.columns = df_SM2miR.columns.str.strip()
	
	for i in df_SM2miR.columns:
		try:
			df_SM2miR[i] = df_SM2miR[i].str.strip()
		except:
			continue

	# chemical information
	df_chemical_SM2miR = pd.read_csv(root + '/data/chemical_information.txt', sep="\t", encoding="ISO-8859-1")
	df_chemical_repoHub = pd.read_csv(root + '/data/chemical_information_repo_hub.txt', sep="\t", encoding="ISO-8859-1")

	df_chemical_SM2miR = df_chemical_SM2miR.sort_values('CID')
	print('Number of small molecules', len(set(df_SM2miR["CID"])))

	return df_SM2miR, df_chemical_SM2miR, df_chemical_repoHub


def dataset_species_stats(df_SM2miR, root_dir):
  '''
  Count number of miRNA per species
  '''
  
  species = set(df_SM2miR['Species'])
  count_miR = np.zeros(shape = (len(species), len(species)))
  for idx,i in enumerate(species):
    df_i = df_SM2miR[df_SM2miR['Species'] == i]
    for idy,j in enumerate(species):   

      df_j = df_SM2miR[df_SM2miR['Species'] == j]
      count_miR[idx,idy] = len(set(df_i['miRBase']) & set(df_j['miRBase']))


  fig=plt.figure(figsize=(3,3), dpi= 100, facecolor='w', edgecolor='k')
  df_count = pd.DataFrame(count_miR, columns = species, index = species)
  ax = sns.heatmap(df_count, annot=True, cmap="YlGnBu",  annot_kws={"fontsize":7}, fmt='g')

  fig=plt.figure(figsize=(10,5))
  c = np.diag(count_miR)
  idx_sorted = np.flip(np.argsort(c))
  species_sorted = list(species)
  species_sorted = [species_sorted[i] for i in idx_sorted]

  plt.bar(range(1,len(c)+1), c[idx_sorted])
  plt.ylabel('Number of small molecules-miRNA associations')
  plt.xticks(range(1,len(c)+1), species_sorted, rotation ='vertical')
  plt.yscale('log')
  
  return count_miR

def get_matrices(df_SM2miR, df_chemical_SM2miR, species_chosen, chemFeature, df_repoHub):
  '''
  Y: contains binary associations small molecules x miRNAs
  F: features for small molecules.
  '''
  df_SM2miR_organim = df_SM2miR[df_SM2miR['Species'] == species_chosen].copy() # choose species
  df_SM2miR_organim["value"] = 1
  df_Y = pd.pivot_table(df_SM2miR_organim, values="value", index=["CID"], columns="miRBase", fill_value=0) 
  del df_Y['Dead miRNA entry']
  Y = df_Y.values
   
  # filter per organism
  df_chemical_SM2miR_tmp = df_chemical_SM2miR[df_chemical_SM2miR['CID'].isin(list(df_Y.index))]
 
  if chemFeature == 'MACCS':
    F = df_chemical_SM2miR_tmp['MACCS_FP'].values
    X = str2np(F)
    XrepoHub = str2np(df_repoHub['MACCS_FP'].values)
  elif chemFeature == 'RDKit':
    F = df_chemical_SM2miR_tmp['RDKit_FP'].values
    X = str2np(F)
  elif chemFeature == 'PubChem':
    F = df_chemical_SM2miR_tmp['CACTVS_FP'].values
    X = str2np(F)
  elif chemFeature == 'standard':
    F = df_chemical_SM2miR_tmp[['molecular_weight', 'atom_stereo_count', 'bond_stereo_count', 'h_bond_acceptor_count', 'h_bond_donor_count', 'tpsa', 'xlogp', 'heavy_atom_count', 'exact_mass', 'count_rings', 'count_aromatic_rings']].values
    F[np.isnan(F)] = 0
    X = zscore(F)

  elif chemFeature == 'random':
    X =  np.random.rand(167,10)

  return Y, X, df_Y,XrepoHub

def str2np(F):
  # convert string to numpy matrix
  X = np.zeros(shape=(F.shape[0], len(F[0])))
  for i in range(F.shape[0]): 
    X[i,:] = np.array(list(F[i]), dtype=int)

  return X

def filter_data(X, Y, df_Y, XrepoHub, plotornot = False):
  '''
  Keep a minimun of 5 association per miRNA
  > 0 per SM.
  '''
  # filtering columns of Y to keep only miRNA with 5 associations
  idy = np.sum(Y, axis = 0) > 4

  Y_filtered = Y[:, idy]
  df_YFiltered = df_Y.loc[:, idy]

  # filtering rows of Y to keep only SM with 2 associations
  idx = np.sum(Y_filtered, axis = 1) > 0

  Y_filtered = Y_filtered[idx, :]
  X_filtered = X[idx, :]
  df_YFiltered = df_YFiltered.loc[idx, :]
  
  # filtering columns of chem FP to keep those > 0
  #idy = np.sum(X_filtered, axis = 0) > 0

  #X_filtered = X_filtered[:, idy]
  #X_repoHubFiltered = XrepoHub[:, idy]
  X_repoHubFiltered = XrepoHub
  print('SMxmiRNAs = ',Y_filtered.shape, 'SMxchemFeat =', X_filtered.shape, 'min per SM =', min(np.sum(Y_filtered, axis = 1)), 'min per miRNA =', min(np.sum(Y_filtered, axis = 0)))

  if plotornot:
    fig=plt.figure(figsize=(15,5), facecolor='w', edgecolor='k')

    ax = fig.add_subplot(1, 2, 1)
    plt.bar(list(range(1, Y_filtered.shape[0] + 1)), sorted(np.sum(Y_filtered, axis = 1), reverse=True), edgecolor='gray', color='gray')
    #ax.set_yscale('log')
    plt.ylabel('Number of miRNA targets')
    plt.xlabel('Active Small molecules (ordered)')

    ax = fig.add_subplot(1, 2, 2)
    plt.bar(list(range(1, Y_filtered.shape[1] + 1)), sorted(np.sum(Y_filtered, axis = 0), reverse=True), edgecolor='gray', color='gray')
    #ax.set_yscale('log')

    plt.ylabel('Number of active small molecules')
    plt.xlabel('miRNA targets (ordered)')
    
  return Y_filtered, X_filtered, df_YFiltered, X_repoHubFiltered


def save_result(root_dir,filename, a):
  import pickle
    
  with open(root_dir + '/data/'+filename+'.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return


def read_result(root_dir,filename):
  import pickle    
  with open(root_dir + '/data/'+filename+'.pickle', 'rb') as handle:
    x = pickle.load(handle)
  return x