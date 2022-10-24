"""
2D dataset with 2 classes for Logistic Regression with Polynomial Expansion
plus: imbalanced binary dataset analysis (recall, precision, ROC, etc)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def data(m=100, shuffle=True):
    n = 2
    mx = np.random.normal(loc=0, scale=(1.5,1), size=(m,n))
    
    #rotate
    angle = np.deg2rad(45)   # 10 degrees
    T = [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]
    mx = np.matmul(T, mx.T).T
    
    #scale
    ranges = mx.max(0) - mx.min(0)
    mx = (mx - mx.min(axis=0)) / ranges
    
    #parabolic curve as class separator
    f = lambda x : 0.63 - (0.7-1.5*x)**2
    xx,yy = mx.T
    y = mask = yy > f(xx)
    
    #push the positives a little bit upwards to form a narrow gap between the classes
    mx[mask,1] += 0.05   
    
    #add errors
    errors = np.random.normal(loc=0, scale=np.std(mx, axis=0)/4, size=mx.shape)
    mx += errors
    
    #final scaling
    ranges = mx.max(0) - mx.min(0)
    mx = (mx - mx.min(axis=0)) / ranges
    
    #shuffle
    if shuffle:
        nx = np.random.permutation(len(mx))
        mx,y = mx[nx], y[nx]
    return(mx,y)

#====================================================================================

X,y = data(m=500)

#visualize
plt.figure()
cmap = ListedColormap(["lightgreen", "m"])
plt.scatter(*X.T, edgecolor="k", s=25,  c=y, cmap=cmap)

#model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

#try degrees 1,2,3,4, etc.
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
sc = StandardScaler()
md = LogisticRegression(solver="liblinear", C=1E9)
pl = make_pipeline(poly, sc, md)
pl.fit(X,y)
ypred = pl.predict(X)

#divide the domain space with the decision boundry and fill 
r = np.linspace(-0.1, 1.1, 100)
XX,YY = np.meshgrid(r,r)
XXX = np.c_[XX.ravel(), YY.ravel()]
ZZ = pl.predict(XXX).reshape(XX.shape)
plt.contourf(XX,YY,ZZ, zorder=-3, cmap=cmap, alpha=0.5, levels=1)


#analize
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, precision_recall_curve, roc_curve, roc_auc_score
accuracy = pl.score(X,y);        print("accuracy =", accuracy.round(2))
mx = confusion_matrix(y, ypred); print("\nconfusion matrix:", mx, sep="\n", end="\n"*2)
s = classification_report(y, ypred, labels=[0,1]);  print(s)

"""CURVES"""
from sklearn.model_selection import cross_val_predict
zz = cross_val_predict(pl, X,y, cv=5, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y, zz)
precisions, recalls = (a[:-1] for a in (precisions, recalls))

#threshold curve
fig = plt.figure(figsize=(10,7));  fig.subplots_adjust(hspace=0.35); fig.add_subplot(221)
plt.plot(thresholds, recalls, 'r-', label='recall')
plt.plot(thresholds, precisions, 'b-', label='precision')
plt.xlabel("z (score)")
plt.legend()

#PR-curve
ppred = cross_val_predict(pl, X,y, cv=5, method="predict_proba")[:,1]
precisions, recalls, thresholds = precision_recall_curve(y, ppred)
precisions, recalls = (a[:-1] for a in (precisions, recalls))

fig.add_subplot(222)
plt.plot(precisions, recalls, label="PR-curve")
plt.xlabel("precision"); plt.ylabel("recall")
plt.legend()


#ROC-curve
FPR, TPR, thresholds = roc_curve(y, ppred)
AUC = roc_auc_score(y, ppred)
fig.add_subplot(223)
plt.plot(FPR, TPR, label="ROC-curve")
plt.xlabel("False Positives Rate"); plt.ylabel("True Positives Rate")
plt.text(0.2, 0.7, f"AUC = {AUC.round(2)}")
plt.plot([0,1], [0,1], '--', color='gray')
plt.legend()
