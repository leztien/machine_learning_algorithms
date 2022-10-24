"""
knn algorithm (as a function)
"""

import numpy as np
import make_data_for_classification   # leztien/synthetic_datasets/make_data_for_classification.py

def split(X,y):
    nx = np.random.permutation(len(X))
    X,y = (nd[nx] for nd in (X,y))
    split = int(len(X)*0.8)
    Xtrain,ytrain = X[:split], y[:split]
    Xtest, ytest = X[split:], y[split:]
    return(Xtrain,Xtest,ytrain,ytest)
 
    
def knn(Xtrain, ytrain, Xtest, ytest=None, nearest_neighbours=3):
    mx = np.square(Xtrain[None,...] - Xtest[:,None,:]).sum(axis=-1).T
    nx = np.argpartition(mx, axis=0, kth=nearest_neighbours)[:nearest_neighbours].T
    Ypred = np.array(ytrain, dtype='uint8').take(nx)
    ypred = np.array([np.bincount(a, minlength=len(set(ytrain))) for a in Ypred], dtype='uint').argmax(axis=1)
    
    #same in one line:
    ypred = np.array([np.bincount(a, minlength=len(set(ytrain))) for a in np.array(ytrain, dtype='uint8').take(np.argpartition(np.square(Xtrain[None,...] - Xtest[:,None,:]).sum(axis=-1).T, axis=0, kth=nearest_neighbours)[:nearest_neighbours].T)], dtype='uint').argmax(axis=1)
    
    if ytest is not None:
        accuracy = np.equal(ytest, ypred).mean()
        print("test set accuracy =", accuracy)
    return ypred

#=============================================================================================
    
X,y = make_data_for_classification(m=500, n=7, k=5, blobs_density=0.5)
Xtrain,Xtest,ytrain,ytest = split(X,y)

ypred = knn(Xtrain, ytrain, Xtest, nearest_neighbours=5)
accuracy = np.equal(ytest, ypred).mean()
print(accuracy)
