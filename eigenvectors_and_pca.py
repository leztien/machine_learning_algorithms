

"""
visualize PCA in action
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import det, inv, eig
from numpy import cov


def make_data(m=500, n=3, centered=False):
    from numpy import array
    from numpy.random import multivariate_normal
    
    Σ = [[9,3,4],
         [0,4,2],
         [0,0,3]]
    Σ = array(Σ, dtype=int)
    Σ = Σ | Σ.T
    X = multivariate_normal(mean=(0,0,0), cov=Σ, size=m)
    if centered:
        μ = X.mean(axis=0)
        X = X - μ
    else:
        X += X.min(axis=0).__abs__()
    return X

#===========================================================

X = make_data(500, centered=True)

Σ = cov(X.T)
λ,E = eig(Σ)  #E = unit eigenvectors arranged vertically
λ = λ.real
Xpca = (E.T @ X.T).T


#VISUALIZATION
fig = plt.figure()
sp = fig.add_subplot(111, projection="3d")
sp.plot(*X.T, '.', ms=2)
sp.plot(*Xpca.T, '.', ms=2)

lim = max(*sp.get_xlim3d(), *sp.get_ylim3d(), *sp.get_zlim3d())
sp.plot((-lim,lim), (0,0), (0,0), color='gray')
sp.plot((0,0), (-lim,lim), (0,0), color='gray')
sp.plot((0,0), (0,0), (-lim,lim), color='gray')
sp.text(lim, 0, 0, "x"); sp.text(0, lim, 0, "y"); sp.text(0,0,lim, "z")
sp.set_xlabel("x", weight="bold"); sp.set_ylabel("y", weight="bold"); sp.set_zlabel("z", weight="bold")

plt.show()
