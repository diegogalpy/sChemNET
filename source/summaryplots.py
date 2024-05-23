import pandas as pd 
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from collections import defaultdict


__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '16-08-2021'



def plot_summary_testset( results):
  recall = defaultdict(list)

  for i in results['recall_test']:  
    for v in i['x']:
      recall['x'].append(v)
      recall['method'].append('our method')
    for v in i['y']:
      recall['y'].append(v)
      

  for i in results['recall_random']:      
    for v in i['x']:
      recall['x'].append(v)
      recall['method'].append('random')
    for v in i['y']:
      recall['y'].append(v)

  

  for i in results['recall_chemsim']:      
    for v in i['x']:
      recall['x'].append(v)
      recall['method'].append('ChemSIM')
    for v in i['y']:
      recall['y'].append(v)
  

  recall_df = pd.DataFrame(recall)
  #recall_df = recall_df[recall_df['x'] <= 1000]
  ndf = recall_df.groupby(['x', 'method']).agg({'x':'mean','y':'mean', 'method': 'first'})

  fig, axes = plt.subplots(1, 3, figsize=(30,7))
  my_pal = {"random": "#2166ac", "our method": "#b2182b"}

  sns.violinplot(y = results['auc_test'], ax=axes[0], color="#b2182b")
  sns.swarmplot(y = results['auc_test'], color=".25", ax=axes[0])
  axes[0].set_ylabel('AUROC per miRNA (test set)')

  sns.scatterplot(x = results['nposlabels'], y = results['auc_test'], ax=axes[1], color ="#b2182b")

  #for i, k in enumerate(targets):
  #    axes[1].annotate(name2target[k], (resultB_exp2['nposlabels'][i], resultB_exp2['auc_test'][i]))
  axes[1].set_xlabel('Number of positive labels')
  axes[1].set_ylabel('AUROC per miRNA (test set)')

  sns.barplot(data = ndf, x = 'x',y  = 'y', hue = 'method', ax=axes[2])
  axes[2].set_xlabel('top-K small molecules retrieved')
  axes[2].set_ylabel('mean recall')

  fig.suptitle('Performance on test set (2,000 small molecules)')
  return 