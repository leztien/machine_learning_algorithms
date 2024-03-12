
"""
Make a plot that enables you to compare visually
how different the erros that your models make.
Based on the 'missingno' library
"""

import warnings
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def plot_errors(X, y, X_val=None, y_val=None, estimators=[], scorer=None, 
                figsize=None, random_state=None):
    """
    Make a plot that enables you to compare visually
    how different the erros that your models make.
    Based on the 'missingno' library
    TODO: better docs
    """
    figsize = figsize or (10, 6)
    test_size = 0.2
    if X_val is None or y_val is None:
        X, X_val, y, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        estimators = [est.fit(X, y) for est in estimators]
        y_preds = [est.predict(X_val) for est in estimators]
        scores = [scorer(y_val, y_pred) for y_pred in y_preds]
    
    # Estimator names
    names = [(est[-1] if hasattr(est, 'steps')  else est).__class__.__name__ 
             for est in estimators]
    
    # What if there are duplicate names of estimators
    counter = Counter()
    for i,e in enumerate(names):
        counter.update([e])
        n = counter.get(e)
        names[i] = f"{e}{n if n > 1 else ''}"
    
    # Need more comments below
    df = pd.DataFrame(y_preds, index=names).T
    df = df.iloc[:, np.argsort(scores)[::-1]]
    df['actual'] = np.array(y_val)
    df.sort_values(['actual'] + list(df.columns[:-1]), inplace=True)
    class_borders = pd.DataFrame(Counter(y_val).items()).sort_values(0).cumsum().iloc[:-1, -1].values
    df = pd.DataFrame(df.values == df[['actual']].values, columns=df.columns).drop('actual', axis=1).replace({False: np.nan})
    ax = msno.matrix(df, figsize=figsize)
    ax.hlines(y=class_borders, xmin=-0.5, xmax=len(estimators)-0.5, color='b')
    [ax.text(i, len(y_val)*1.05, round(s,2)) for i,s in enumerate(sorted(scores, reverse=True))]
    ax.set_title("Errors made by models.\ngrey=correct, white=false predictions\nthe horizontal lines separate the classes")
    ax.set_yticks([])
    return ax


def make_data(m, n, k, random_state=None):
    rs = np.random.RandomState(random_state)
    X = rs.binomial(n=range(n, 0, -1), p=rs.random(n), size=(m,n))
    X = X + rs.normal(loc=0, scale=X.std(axis=0)/(rs.random(n)*4+1), size=(m,n))
    y = (X.sum(axis=1).round() % k).astype(int)
    return X,y



# DEMO
if __name__ == '__main__':
    random_state = None
    X,y = make_data(50, 5, k=3, random_state=None)
    
    md1 = GaussianNB()
    md2 = LogisticRegression()
    md3 = RandomForestClassifier(2, random_state=42)
    md4 = RandomForestClassifier(3, random_state=42)
    
    if np.random.random() > 0.5:
        print("input: dataframe")
        X,y = pd.DataFrame(X), pd.Series(y)
    else:
        print("input: ndarray")
    
    plot_errors(X,y, estimators=[md1, md2, md3, md4], scorer=accuracy_score)
    

