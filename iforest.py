
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import random
import pandas as pd


def c(n):
    if n < 2:
        return 0
    elif n == 2:
        return 1
    else:
        return (2*(np.log(n - 1) + 0.5772156649)) - (2*(n - 1)/n)


class IsolationTreeNode:
    def __init__(self, is_external=None, size=None, attribute=None, split_value=None, left=None, right=None):
        self.is_external = is_external
        self.attribute = attribute
        self.split_value = split_value
        self.right = right
        self.left = left
        self.size = size
        self.adjustment = None
        if self.is_external:
            self.adjustment = c(self.size)


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.height_limit = np.ceil(np.log2(self.sample_size))
        self.n_trees = n_trees
        self.trees = [IsolationTree(self.sample_size, self.height_limit) for _ in range(self.n_trees)]
        self.c = c(sample_size)

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for tree in self.trees:
            tree.fit(X, improved)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        result = np.empty(shape=[X.shape[0], 1])
        for i in range(X.shape[0]):
            row = X[i, :]
            result[i, 0] = sum([tree.get_path_length(tree.root, length=0, row=row) for tree in self.trees])/self.n_trees

        return result

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        path_lengths = self.path_length(X)
        scores = np.empty(shape=[X.shape[0]])
        for i in range(X.shape[0]):
            scores[i] = 2 ** (-path_lengths[i, 0]/self.c)
        return scores

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = np.empty(shape=[scores.shape[0]])
        for i in range(scores.shape[0]):
            predictions[i] = 1 if scores[i] >= threshold else 0
        return predictions

    def predict(self, X:np.ndarray, threshold: float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        return predict_from_anomaly_scores(anomaly_score(X), threshold)


class IsolationTree:
    def __init__(self, sample_size, height_limit):
        self.height_limit = height_limit
        self.sample_size = sample_size
        self.root = None
        self.n_nodes = 0

    def create(self, X:np.ndarray, height) -> IsolationTreeNode:
        self.n_nodes = self.n_nodes + 1
        if X is None:
            return IsolationTreeNode(is_external=True, size=0)
        if X.shape[0] <= 1 or height >= self.height_limit:
            return IsolationTreeNode(is_external=True, size=X.shape[0])

        n_columns = X.shape[1]
        attribute = random.randint(0, n_columns - 1)
        att_min = np.min(X[:, attribute])
        att_max = np.max(X[:, attribute])
        if att_min == att_max:
            return IsolationTreeNode(is_external=True, size=X.shape[0])

        # split_value = random.uniform(att_min, att_max)
        split_value = random.betavariate(0.5, 0.5)*(att_max - att_min) + att_min
        smaller = X[X[:, attribute] < split_value, :]
        greater = X[X[:, attribute] >= split_value, :]
        node = IsolationTreeNode(is_external=False, attribute=attribute, split_value=split_value)
        node.left = self.create(smaller, height + 1)
        node.right = self.create(greater, height + 1)
        return node

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        sample_ids = random.sample(range(X.shape[0]), self.sample_size)
        sample = X[sample_ids]
        self.root = self.create(sample, 0)
        return self.root

    def get_path_length(self, node: IsolationTreeNode, length, row):
        if node is None:
            return length
        if node.is_external:
            return length + node.adjustment

        if row[node.attribute] <= node.split_value:
            return self.get_path_length(node.left, length + 1, row)
        else:
            return self.get_path_length(node.right, length + 1, row)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """

    y = np.array(y)[:]

    threshold = 1.0
    while threshold >= 0:
        n_tp = 0
        n_fp = 0
        n_tn = 0
        n_fn = 0
        for i in range(scores.size):
            if y[i] == 1 and scores[i] >= threshold:
                n_tp = n_tp + 1
            elif y[i] == 1 and scores[i] < threshold:
                n_fn = n_fn + 1
            elif y[i] == 0 and scores[i] >= threshold:
                n_fp = n_fp + 1
            elif y[i] == 0 and scores[i] < threshold:
                n_tn = n_tn + 1
        tpr = n_tp/(n_tp + n_fn)
        if tpr >= desired_TPR:
            fpr = n_fp/(n_fp + n_tn)
            return threshold, fpr
        threshold = threshold - 0.001
