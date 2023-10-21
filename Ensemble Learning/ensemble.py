from ID3 import DecisionTree, RandomForestTree
import numpy as np
import multiprocessing as mp
from math import log, exp
from statistics import mode
import random

class AdaBoost:
    def __init__(self):
        self.stumps = list
        self.sample_weights = list
        self.stump_say = list
    
    def error(self, pred: list, target: list):
        assert len(pred) == len(target)
        err = 0
        for i in range(len(pred)):
            if pred[i] != target[i]: err += self.sample_weights[i]
        return err

    def train(self, data, epochs: int=100):
        self.sample_weights = [1/len(data)] * len(data)
        self.stumps = [None] * epochs
        self.stump_say = [None] * epochs

        for i in range(epochs):
            self.stumps[i] = DecisionTree() 
            self.stumps[i].makeTree(data=data, weights=list(self.sample_weights), max_depth=1)

            pred = []
            for d in data:
                pred.append(self.stumps[i].predict(d))
            err = self.error(pred, [d['label'] for d in data])   
            # print(f"tree error: {err}")

            self.stump_say[i] = 0.5*log((1 - err) / err)

            for j in range(len(self.sample_weights)):
                v = 1 if data[j]['label'] == self.stumps[i].predict(data[j]) else -1
                self.sample_weights[j] = self.sample_weights[j] * exp(-self.stump_say[i] * v)

            self.sample_weights = np.divide(self.sample_weights, np.sum(self.sample_weights))
    
    def predict(self, data, true_false_values = ('yes' ,'no')):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            Hx = 0
            for j, stump in enumerate(self.stumps):
                Hx += self.stump_say[j]*(1 if stump.predict(d) == true_false_values[0] else -1)
            pred[i] = true_false_values[0] if np.sign(Hx) == 1 else true_false_values[1]
        
        return pred


def bagAndMakeTree(data, num_samples):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = DecisionTree()
    tree.makeTree(bag)
    return tree

class BaggedTrees:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_workers = None):
        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(bagAndMakeTree, zip(mult_data, mult_samp))

    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers = 4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree : tree.predict(d), self.trees))

        return pred

def rfBagTree(data, num_samples, num_attributes):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = RandomForestTree()
    tree.makeTree(bag, num_attributes=num_attributes)
    return tree

class RandomForest:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_attributes: int = 4, num_workers = None):

        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees
        mult_attr = [num_attributes] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(rfBagTree, zip(mult_data, mult_samp, mult_attr))
            
    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers = 4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree : tree.predict(d), self.trees))

        return pred
