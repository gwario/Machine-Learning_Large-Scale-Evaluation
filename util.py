import logging as log
import pprint as pp

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def generate_test_sets(X_test, y_test, repetition_i, train_size, config):

    log.debug("# TEST_SETS: {}".format(config['test_splits']))
    log.debug("TEST_SIZE: {}".format((1 - train_size) / config['test_splits']))

    X_test_sets = []
    y_test_sets = []

    X_remaining = X_test
    y_remaining = y_test
    for test_i in range(config['test_splits'] - 1):
        actual_test_size = 1 / (config['test_splits'] - test_i)

        X_rest, X_test_i, y_rest, y_test_i = train_test_split(X_remaining, y_remaining,
                                                              stratify=y_remaining if config['stratify'] else None,
                                                              test_size=actual_test_size,
                                                              random_state=config['random_state']+repetition_i)
        X_test_sets.append(X_test_i)
        y_test_sets.append(y_test_i)

        X_remaining = X_rest
        y_remaining = y_rest
        log.debug("TEST_SET_{}:\n{}".format(test_i, pp.pformat(X_test_i)))
        log.debug("Remaining after TEST_SET_{}:\n{}".format(test_i, pp.pformat(X_remaining)))

    X_test_sets.append(X_remaining)
    y_test_sets.append(y_remaining)
    log.debug("TEST_SET_{}:\n{}".format(config['test_splits'] - 1, pp.pformat(X_remaining)))
    return X_test_sets, y_test_sets


def get_estimator(estimator_config):

    estimator_name = estimator_config['estimator']
    estimator_params = estimator_config['params']

    if estimator_name == 'RandomForestClassifier':
        estimator = RandomForestClassifier()
    elif estimator_name == 'ExtraTreesClassifier':
        estimator = ExtraTreesClassifier()
    elif estimator_name == 'AdaBoostClassifier':
        estimator = AdaBoostClassifier()
    elif estimator_name == 'DecisionTreeClassifier':
        estimator = DecisionTreeClassifier()
    elif estimator_name == 'GaussianNB':
        estimator = GaussianNB()
    elif estimator_name == 'KNeighborsClassifier':
        estimator = KNeighborsClassifier()
    elif estimator_name == 'SVC':
        estimator = SVC()
    elif estimator_name == 'GaussianProcessClassifier':
        estimator = GaussianProcessClassifier()
    elif estimator_name == 'MLPClassifier':
        estimator = MLPClassifier()
    elif estimator_name == 'Perceptron':
        estimator = Perceptron()
    else:
        raise ValueError("Unknown estimator '{}'".format(estimator_name))

    estimator.set_params(**estimator_params)
    estimator_config['params'] = estimator.get_params()

    return estimator


class StrDateToUnixTs(BaseEstimator, TransformerMixin):
    """Prints the input."""

    def __init__(self,label = None):
        self.label = label

    def fit(self, x, y=None):
        return self

    def transform(self, input):

        input['dteday'] = pd.to_numeric(pd.to_datetime(input['dteday']))

        return input


def get_search_space(estimator):
    """Returs the search space for each classifier. If estimator.get_params() has a random_state,
    that value is also used for the search space."""

    estimator_name = estimator.__class__.__name__

    if estimator_name == 'RandomForestClassifier' or estimator_name == 'ExtraTreesClassifier':
        search_space = {
            'random_state': [estimator.get_params()['random_state']],
            'n_estimators': [10, 20, 50, 100, 200, 500],
            'max_depth':    [5, 5, 10, 13, 15, 17, 20, 25, 30, 40, None],
            'max_features': ['sqrt', 'log2', 0.33, 0.5, 0.25, None],
            'criterion':    ["gini", "entropy"],
            'min_samples_split':[2, 3, 4, 5, 0.01, 0.05, 0.001, 0.005],
            'min_samples_leaf': [1, 3, 5, 7, 0.001, 0.005, 0.01, 0.05, 0.0001, 0.0005],
            'max_leaf_nodes':   [None, 5, 10, 15, 20, 35, 30, 35, 40, 45, 50]
        }
        n_iter = 50
    elif estimator_name == 'AdaBoostClassifier':
        search_space = {
            'random_state': [estimator.get_params()['random_state']],
            'n_estimators': [30, 40, 50, 70, 90],
            'learning_rate':[0.5, 0.65, 0.75, 0.85, 0.9, 1],
            'algorithm':    ['SAMME', 'SAMME.R']
        }
        n_iter = 50
    elif estimator_name == 'DecisionTreeClassifier':
        search_space = {
            'random_state': [estimator.get_params()['random_state']],
            'splitter':     ["best", "random"],
            'max_depth':    [5, 5, 10, 13, 15, 17, 20, 25, 30, 40, None],
            'max_features': ['sqrt', 'log2', 0.33, 0.5, 0.25, None],
            'criterion':    ["gini", "entropy"],
            'min_samples_split':[2, 3, 4, 5, 0.01, 0.05, 0.001, 0.005],
            'min_samples_leaf': [1, 3, 5, 7, 0.001, 0.005, 0.01, 0.05, 0.0001, 0.0005],
            'max_leaf_nodes':   [None, 5, 10, 15, 20, 35, 30, 35, 40, 45, 50]
        }
        n_iter = 2000
    elif estimator_name == 'GaussianNB':
        search_space = None
        n_iter = -1
    elif estimator_name == 'KNeighborsClassifier':
        search_space = {
            'algorithm':    ['ball_tree', 'kd_tree', 'brute'],
            'n_neighbors':  [1, 2, 3, 4, 7, 9],
            'metric':       ['euclidean', 'manhattan', 'cityblock', 'l1', 'l2'],
            'weights':      ["uniform", "distance"]
        }
        n_iter = 30
    elif estimator_name == 'SVC':
        search_space = {
            'random_state': [estimator.get_params()['random_state']],
            'C':            [1, 10, 100, 200, 500],
            #'kernel':       ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'degree':       [2, 3, 4, 5, 6, 7, 8, 9],
            'gamma':        [0.05,'auto',0.1,0.001,0.15,0.2,0.3],
            'coef0':        [0.0, 1.0, 0.5, 4.2, 2.0, 0.73],
            'tol':          [0.01,0.00001,1,10,50,100],
            'max_iter':     [-1, 50000],
        }
        n_iter = 30
    elif estimator_name == 'GaussianProcessClassifier':
        search_space = {
            'random_state':         [estimator.get_params()['random_state']],
            'n_restarts_optimizer': [2, 3, 5, 7, 8, 12],
            'max_iter_predict':     [80, 100, 150, 200],
            'multi_class':          ['one_vs_rest', 'one_vs_one'],
        }
        n_iter = 15
    elif estimator_name == 'MLPClassifier':
        search_space = None #TODO define MLPClassifier search space
        #search_space = {
        #    'random_state': [estimator.get_params()['random_state']]
        #}
        n_iter = 20
    elif estimator_name == 'Perceptron':
        search_space = {
            'random_state': [estimator.get_params()['random_state']],
            'penalty': [None, 'l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.0005, 0.001, 0.005,],
            'fit_intercept': [True, False],
            'max_iter': [5, 10, 50, 100, 500, 1000, 5000, 8000, 10000],
            'tol': [1e-3, None],
            'shuffle': [True, False],
            'eta0': [1, 0.5, 1.5],
        }
        n_iter = 7
    else:
        raise ValueError("Unknown estimator '{}'".format(estimator_name))

    return search_space, n_iter


def timedelta_milliseconds(td):
    return td.days * 86400000 + td.seconds * 1000 + td.microseconds / 1000
