"""
Markov Model for text generation
"""

import re
from collections import Counter
from itertools import groupby
from random import random


# Make sure the path is correct
PATH = r"/home/linux-ubuntu/Datasets/Bible.txt"
PATH = r"Bible.txt"


def tokenize(path):
    with open(PATH, mode='r', encoding='utf-8') as f:
        text = f.read()
    pattern = re.compile(r"[0-9\*\[\]\(\)\s]+")
    text = pattern.sub(' ', text).replace(chr(160), '')
    return text.split()  #list


def ngramize(seq, n=3):
    assert n>=2, "n must be greater than 1"
    return [tuple(seq[i+j] for j in range(n)) for i in range(0, len(seq)-(n-1))]


def compute_distributions(ngrams):
    counter = Counter(ngrams)
    g = groupby(sorted(counter.items(), key=lambda t: t[0]), key=lambda t: t[0][:-1])
    distributions = dict()
    for key, gen in g:
        values, counts = zip(*sorted(((e[0][-1], e[-1]) for e in gen), key=lambda t: t[0]))
        total = sum(counts)
        probs = tuple(count/total for count in counts)
        distributions[key] = {'values': values, 'probs': probs}
    return distributions


def sample_distribution(distributions, key:'ngram'):
    values = distributions[key]['values']
    probs = distributions[key]['probs']
    r = random()
    cum = 0.0
    for v,p in zip(values, probs):
        cum += p
        if cum > r:
            return v
    return v


def markov_chain(initial_state, length, distributions):
    state = tuple(initial_state)
    l = list(state)
    for _ in range(length):
        word = sample_distribution(distributions, key=state)
        l.append(word)
        state = tuple(l[-len(state):])
    return str.join(' ', l)


##############################################################################

tokens = tokenize(PATH)
ngrams = ngramize(tokens, 3)
distributions = compute_distributions(ngrams)

start = ("и", "сказал")
length = 500

generated_text = markov_chain(initial_state=start, 
                              length=length, 
                              distributions=distributions)
print(generated_text)
