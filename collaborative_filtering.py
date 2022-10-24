"""
SDV-inspired matrix completion via gradient descent
"""

import numpy as np
from numpy.linalg import svd
from numpy import dot


def make_matrix(m, n, k, p=None, max_value=None):    
    p = p or 0.1
    mx = max_value or min((m+n)//3, 5)
    U = np.random.randint(0,mx, size=(m,k))
    V = np.random.randint(0,mx, size=(n,k))    
    Y = np.matmul(U, np.transpose(V)).astype('float')
    nx = np.random.permutation(Y.size)[:int(Y.size*p)]
    Y.ravel()[nx] = np.nan
    return Y



#data
m,n,k = 20,7,3
Y = make_matrix(m, n, k, p=0.1)


#utils
def frobenius_cost(Y, U, V):
    return np.nansum((U@V.T - Y)**2)**0.5


def get_u_gradients(Y,U,V):
    U_gradients = np.zeros_like(U)
    m,n = Y.shape
    for i in range(m):
        for j in range(n):
            if np.isnan(Y[i,j]): continue
            U_gradients[i] += (dot(U[i],V[j]) - Y[i,j]) * V[j] + λ*U[i]
    return U_gradients


def get_v_gradients(Y,U,V):
    V_gradients = np.zeros_like(V)
    m,n = Y.shape
    for j in range(n):
        for i in range(m):
            if np.isnan(Y[i,j]): continue
            V_gradients[j] += (dot(U[i],V[j]) - Y[i,j]) * U[i] + λ*V[j]
    return V_gradients



#hyperparameters
η = 0.01
λ = 0.001


#initialize 
U = np.random.random(size=(m,k)) / 100
V = np.random.random(size=(n,k)) / 100


#loop
for epoch in range(500):
    U_gradients = get_u_gradients(Y, U, V)
    U -= η*U_gradients
    
    V_gradients = get_v_gradients(Y, U, V)
    V -= η*V_gradients
    
    #cost
    if epoch%10==0:
        print(frobenius_cost(Y, U, V))


#predict
Y_pred = U@V.T
print(Y, Y_pred.round(1), (Y == Y_pred.round(1)), sep="\n\n")
