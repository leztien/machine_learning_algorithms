
"""
Apriori algorithm
(quick and dirty)
"""


import numpy as np
from pandas import DataFrame
from math import factorial


# make data / df
m,n = 10, 5
p = 0.2
X = np.random.choice([0,1], size=(m,n), replace=True, p=(1.0-p, p)).astype(np.int8)
df = DataFrame(X, columns=[chr(97+j) for j in range(n)], index=[f"t{i+1}" for i in range(m)])



def compute_support(df, min_support=None):
    """
    computes support for all possible itemsets 
    (above the predefined min_support)

    assumes the `df`is a DataFrame of boolean values
    with the columns representing items
    """

    # initial checks and preparation
    min_support = float(min_support or 0)
    assert 0 <= min_support < 1, "min support must be in [0,1]"
    m = len(df)
    itemsets = list()

    # recurse
    def recurse(left, right):
        if not right:
            return

        for i,c in enumerate(right):
            current = left + [c]

            #compute support
            support = df[current].all(axis=1).sum() / len(df)
            if support < min_support:
                continue

            itemsets.append((current, support))
            recurse(current, right[i+1:])

    recurse(left=[], right=sorted(df.columns))
    return {frozenset(itemset):support for itemset, support in itemsets}


def make_rules(array_or_set):
    l = list(array_or_set)
    l = [({l.pop(i)}, set(l)) for i,l in zip(range(len(l)), (l.copy() for _ in range(len(l))))]
    return [(b,a) for a,b in l]



## demo ##
min_support = 0.1   # arbitrary
mapping = itemset_supports = compute_support(df, min_support=min_support)


for itemset, support in itemset_supports.items():
    
    if len(itemset) == 1:
        continue

    for rule in make_rules(itemset):
        antecedent, consequent = rule
        confidence = support / mapping[frozenset(antecedent)]

        print((f"Rule: {antecedent} => {consequent} \t"
               f"Support: {round(support, 2)} \t"
               f"Confidence: {round(confidence, 2)} ".replace("'", "")))
