import numpy.matlib
from numpy import linalg as LA
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

__author__ = 'diegogaleano'
__email__  = 'diegogaleano05@gmail.com'
__date__  = '09-09-2020'

    
def DecompositionAlgorithm(R, k, alpha):

    # Add a parameter to see if the algorithm converged:
    maxiter = 2000
    convergence = False
    tolx = 1e-3
    #maxiter = 2000
    variance = 0.01
    # Get the dimensions
    (ndrugs,nses) = R.shape
    epsilon = np.finfo(float).eps
    sqrteps = np.sqrt(epsilon);
    # initialization
    W0 = np.random.uniform(0,np.sqrt(variance),(ndrugs,k))
    H0 = np.random.uniform(0,np.sqrt(variance),(k,nses)) 
    #     print(H0.shape)
    # normalization
    H0 = np.divide(H0, np.matlib.tile(np.array([np.sqrt(np.sum(np.power(H0,2),1))]).transpose(), (1, nses)))

    CT = R > 0
    UN = R == 0

    #get machine precision eps
    epsilon = np.finfo(float).eps
    sqrteps = np.sqrt(epsilon);
    J = []
    for iteration in range(maxiter):
        numer = np.dot(np.multiply(CT,R),H0.transpose())
        
        W = np.maximum(0,np.multiply(W0,np.divide(numer,np.dot((np.multiply(CT,np.dot(W0,H0)) + np.multiply((alpha*UN),np.dot(W0,H0))),H0.transpose())+np.spacing(numer))))

        # Delete negative values due to machine precision.
        W.clip(min = 0)

        numer = np.dot(W.transpose(),np.multiply(CT,R))

        H = np.maximum(0,np.multiply(H0,np.divide(numer,(np.dot(W.transpose(),np.multiply(CT,W.dot(H0)) + np.multiply(alpha*UN,W.dot(H0))) + np.spacing(numer)))))

        # Delete negative values due to machine precision.
        H.clip(min = 0)

        J.append(0.5*LA.norm(np.multiply(CT,(R - np.dot(W,H))),'fro')**2 + 0.5*alpha*LA.norm(np.multiply(UN,(R-np.dot(W,H))),'fro')**2)


        # Get norm of difference and max change in factors
        dw = np.amax(np.abs(W-W0))/(sqrteps + np.amax(np.abs(W0)));
        dh = np.amax(np.abs(H-H0))/(sqrteps + np.amax(np.abs(H0)));
        delta = np.maximum(dw,dh)
        #if iteration % 100 == 0:
        #    print('Iter', iteration, 'cost function', J[-1])

        # Check for convergence
        if iteration > 1:
            if delta <= tolx:
                print('Iter', iteration, 'delta', delta)
                convergence = True
                break

        # Remember previous iteration results
        W0 = W
        H0 = np.divide(H,np.matlib.tile(np.array([np.sqrt(np.sum(np.power(H0,2),1))]).transpose(), (1, nses))) #normalise

   
    return [W,H,J]

	