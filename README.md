# sChemNET: A deep learning framework for predicting small molecules targeting microRNA function. 

This repository contains a demo on how to run our sChemNET algorithm on the SM2miR dataset that we used in our study.

### Datasets

The folder data contains all the datasets needed. 

1. Sm2miR3_2015.txt. contains the small molecule-miRNA associations from the SM2miR database v2015.
2. mature.fa. contains the mature sequences of miRNAs obtained from the miRBase database.
3. repurposing_drugs and repurposing_samples. contains the snapshot from the Drug Repurposing Hub.
4. chemical_information.txt contains chemical features of the small molecules in SM2miR.
5. chemical_information_repo_hub.txt contains the chemical features of the small molecules in the Drug Repurposing Hub database.
6. best_modelZseqSIM_homosapiens.pickle is a dictionary that contains the optimal hyperparameters to run sChemNET on the Homo sapiens dataset.

### Code

All the source code needed to run the step-by-step demo can be found in the folder source. The demo is self-contained. The dependencies are shown in the demo.

## Bugs and suggestions
If you find any bug in our code, please let us know: dgaleano@ing.una.py.

## References
If you find these resources useful, please cite our work:

Galeano et al. sChemNET: A deep learning framework for predicting small molecules targeting microRNA function, Nature Communications, 2024.

## Research page

https://diegogalpy.github.io/



 
