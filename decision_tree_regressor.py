
"""
decision tree regressor (on continuous features)
"""


from random import randint
from operator import lt, itemgetter
import numpy as np

import os
from urllib.request import urlopen
from tempfile import mkstemp


def get_module_from_github(url):
    """
    Loads a .py module from github (raw)
    Returns a module object
    """
    with urlopen(url) as response:
        if response.code == 200:
            text = str(response.read(), encoding="utf-8")
    
    _, path = mkstemp(suffix=".py", text=True)
    
    with open(path, mode='wt', encoding='utf-8') as fh:
        fh.write(text)
    
    directory, file_name = os.path.split(path)
    working_dir = os.getcwd()
    os.chdir(directory)
    module = __import__(file_name[:-3])
    os.chdir(working_dir)
    os.remove(path)
    return module

#######################################################


class Node:
    def __init__(self, split=None, prediction=None):
        self.left = None
        self.right = None
        self.split = split  # (j,v)
        self.prediction = prediction
    
    def __call__(self, x):
        if self.split is None:
            return self.prediction
        j, v = self.split
        op = lt
        next_node = self.right if op(x[j], v) else self.left
        return next_node(x)
    
    def predict(self, x):
        return self.__call__(x)



def get_best_split(X, y, ix):
    # if less than 2 points then no splitting is necessary
    if len(ix) <= 1:
        return None
    
    splits = [(y[ix].var(), None), ] # (var, ((j,v), ix_left, ix_right))
        
    for j in range(X.shape[1]):
        # define the splitting points
        xx = np.sort(X[ix,j])
        values = (xx[:-1] + xx[1:]) / 2
        op = np.less

        for v in values:
            mask = op(X[ix,j], v)
            
            ix_left = ix[~mask]
            ix_right = ix[mask]
            
            var_left = y[ix_left].var()
            var_right = y[ix_right].var()
            var_weighted = (var_left * (len(ix_left) / len(ix)) 
                          + var_right * (len(ix_right) / len(ix)))
            splits.append( (var_weighted, ((j, v), ix_left, ix_right)) )
    
    # get the split with the min gini
    return min(splits, key=itemgetter(0))[1]
        


def make_tree(X, y, ix=None, max_depth=None, **kwargs):

    # initialize ix (index)
    ix = np.array(range(len(y)), dtype=int) if ix is None else ix
    
    # which depth are you on?
    depth = kwargs.get('depth', 0)
    
    assert len(ix) >= 1, "len(ix) must not be zero" + f" {depth}"
    
    # BASE CASE
    if  len(ix) == 1 or depth == max_depth:
        return Node(prediction=y[ix].mean())
    
    # get the best split
    best_split = get_best_split(X, y, ix)
    
    if best_split is None:
        return Node(prediction=y[ix].mean())
    
    # RECURSIVE CASE
    split, ix_left, ix_right = best_split
    node = Node(split=split)
    node.left = make_tree(X, y, ix_left, max_depth, depth=depth+1)
    node.right = make_tree(X, y, ix_right, max_depth, depth=depth+1)
    return node



# Demo
if __name__ == '__main__':

    # Make data
    url = ("https://raw.githubusercontent.com/leztien/synthetic_data/master"
           "/make_data_for_decision_tree_regressor.py")
    module = get_module_from_github(url)
    make_data = module.make_data_for_decision_tree_regressor
    
    m,n = (randint(100, 2000) // 100) * 100, randint(1, 25)
    max_depth = int(np.log(m*n))
    
    X,y = make_data(m, n, categorical_features_proportion=0)
    print(f"m = {m}\tn = {n}\ntree max depth = {max_depth}")
    
    
    # Sklearn model
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score
    
    md = DecisionTreeRegressor(max_depth=max_depth).fit(X,y)
    rsq = md.score(X,y)
    print("sklearn R^2:", rsq.round(2))
    
    
    # My model
    tree = make_tree(X, y, max_depth=max_depth)
    y_pred = [tree.predict(x) for x in X]
    y_true = y
    rsq = r2_score(y_true, y_pred)
    print("my R^2:", rsq.round(2))
