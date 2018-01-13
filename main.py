#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function, absolute_import

import logging as log
import pprint as pp
import numpy as np
import datetime

import pandas as pd
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate, KFold, train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_regression

import file_io as io
import report as rp

# Display progress logs on stdout
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')

random_state = 12345


def generate_test_sets(X_test, y_test):

    log.debug("# TEST_SETS: {}".format(config['test_splits']))
    log.debug("TEST_SIZE: {}".format((1 - config['train_size']) / config['test_splits']))

    X_test_sets = []
    y_test_sets = []

    X_remaining = X_test
    y_remaining = y_test
    for test_i in range(config['test_splits'] - 1):
        actual_test_size = 1 / (config['test_splits'] - test_i)

        X_rest, X_test_i, y_rest, y_test_i = train_test_split(X_remaining, y_remaining,
                                                              stratify=y_remaining if stratify else None,
                                                              test_size=actual_test_size,
                                                              random_state=random_state+repetition_i)
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


if __name__ == '__main__':

    # Template for config.json
    config = {
        'experiment': 'experiment_1',               # the title of the experiment
        'stratify': True,                           # whether to stratify or not
        'repetitions': 1,                           # number of times every estimator is run with every dataset
        'datasets': ['iris.arff',],
                     #'mammography.arff',
                     #'speeddating.arff'],          # preprocessed dataset
        'train_size': 0.6,                          # size of training set
        'test_splits': 2,                           # the number of test splits (test_size = (1-train_size)/test_spits)
        'estimators': [                             # the estimators
            {
                'estimator': 'ExtraTreesClassifier', # estimator name from the list of available estimators
                'params': {}                        # estimator parameters, see scikit docs
            },
            #{
            #    'estimator': 'ExtraTreesClassifier',
            #    'params': {}
            #},
        ]
    }

    stratify = None
    if config['datasets'] == True:
        stratify = True

    result = [{'dataset': '<filename_0>',
               'results': [{'algorithm': '<algo_name_0>',
                            'iterations': [{'iteration': 0,
                                            'metrics': ['fit_time', 'test_0_time', 'test_1_time',
                                                        'train_accuracy', 'test_0_accuracy', '<test_0_accuracy>',
                                                        '...']},
                                           {'iteration': 1,
                                            'metrics': ['<fit_time>', '<test_0_time>', '<test_1_time>',
                                                        '<train_accuracy>', '<test_0_accuracy>', '<test_1_accuracy>',
                                                        '...']}
                                           ],
                            },
                            {'algorithm': '<algo_name_1>',
                             'iterations': [{'iteration': 0,
                                             'metrics': ['fit_time', 'test_0_time', 'test_1_time',
                                                         'train_accuracy', 'test_0_accuracy', '<test_0_accuracy>',
                                                         '...']},
                                            {'iteration': 1,
                                             'metrics': ['<fit_time>', '<test_0_time>', '<test_1_time>',
                                                         '<train_accuracy>', '<test_0_accuracy>', '<test_1_accuracy>',
                                                         '...']}
                                            ],
                             },
                           ]
               },
              {'dataset': '<filename_0>',
               'results': [{'algorithm': '<algo_name_0>',
                            'iterations': [{'iteration': 0,
                                            'metrics': ['fit_time', 'test_0_time', 'test_1_time',
                                                        'train_accuracy', 'test_0_accuracy', '<test_0_accuracy>',
                                                        '...']},
                                           {'iteration': 1,
                                            'metrics': ['<fit_time>', '<test_0_time>', '<test_1_time>',
                                                        '<train_accuracy>', '<test_0_accuracy>', '<test_1_accuracy>',
                                                        '...']}
                                           ],
                            },
                           {'algorithm': '<algo_name_1>',
                            'iterations': [{'iteration': 0,
                                            'metrics': ['fit_time', 'test_0_time', 'test_1_time',
                                                        'train_accuracy', 'test_0_accuracy', '<test_0_accuracy>',
                                                        '...']},
                                           {'iteration': 1,
                                            'metrics': ['<fit_time>', '<test_0_time>', '<test_1_time>',
                                                        '<train_accuracy>', '<test_0_accuracy>', '<test_1_accuracy>',
                                                        '...']}
                                           ],
                            },
                           ]
               },
              ]

    for dataset_filename in config['datasets']:
        log.info("DATASET: {}".format(dataset_filename))

        # load preprocessed dataset
        X, y = io.load_data(dataset_filename)

        for estimator_i, estimator in enumerate(config['estimators']):
            log.info("ESTIMATOR_{}: {}".format(estimator_i, estimator['estimator']))

            # TODO load algorithm
            estimator = None

            # for every repetition
            for repetition_i in range(config['repetitions']):
                log.info("REPETITION_{}".format(repetition_i))

                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    stratify=y if stratify else None,
                                                                    train_size=config['train_size'],
                                                                    random_state=random_state+repetition_i)
                log.debug("TRAIN_SET:\n{}".format(pp.pformat(X_train)))
                log.debug("TEST_SET:\n{}".format(pp.pformat(X_test)))

                # generate test sets
                X_test_sets, y_test_sets = generate_test_sets(X_test, y_test)

                log.info("TEST_SETS:\n{}".format(pp.pformat(X_test_sets)))

                # TODO cross-validated evaluation with all metrics for X_train and all in X_test_sets

                # TODO store metrics

                # TODO create output directory with config.experiment
                # TODO store entire config (actual estimator params)
                # TODO store results arff
                # TODO store results diagrams

                # TODO arff output template
                # dataset, algorithm, repetition_index, split_index, metric1, metric2, metric3

                # TODO generate diagrams (maybe external script?)

    #mode = 'hpsearch'
    mode = 'evaluate'
    #mode = 'evaluate_rwa'

    selector = SelectKBest(mutual_info_regression, k=10)
    regressor = ExtraTreesRegressor(random_state=random_state)

    ##########################################
    ### hpsearch
    n_iter = 80
    hp_cv = KFold(n_splits=5,
                  shuffle=True,
                  random_state=random_state)
    scoring = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error',
               'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']
    param_space = {  # 4*6*3*2*7*6= 4032 / 100 = 40
        "n_estimators": [50, 100, 500, 900],
        "max_depth": [10, 20, 30, None],
        "max_features": ['sqrt', 'log2', None],
        "criterion": ["mse", "mae"],
        "min_samples_split": [9, 6, 3, 2, 0.2, 0.35, 0.5],
        "min_samples_leaf": [5, 3, 1, 0.2, 0.35, 0.5],
    }
    ##########################################
    ### eval
    eval_cv = KFold(n_splits=10,
                    shuffle=True,
                    random_state=random_state)
    #eval_cv = LeaveOneOut()
    eval_params = {
        'criterion': 'mse',
        'max_depth': None,
        'max_features': 'log2',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 500,
        'random_state': random_state,
        'n_jobs': 4
    }
    ##########################################

    x_w_train, y_w_train_true = io.load_data('winequality-white.csv')
    x_r_train, y_r_train_true = io.load_data('winequality-red.csv')
    x_a_train, y_a_train_true = io.load_data('winequality-all.csv')

    # Select features
    log.info("Feature count {}".format(x_a_train.shape[1]))
    selector = SelectKBest(chi2, k=selector.get_params()['k'])
    x_r_train = select_features(x_r_train, y_r_train_true, selector)
    selector = SelectKBest(chi2, k=selector.get_params()['k'])
    x_w_train = select_features(x_w_train, y_w_train_true, selector)
    selector = SelectKBest(chi2, k=selector.get_params()['k'])
    x_a_train = select_features(x_a_train, y_a_train_true, selector)
    log.info("Feature count {}".format(x_a_train.shape[1]))

    if mode == 'hpsearch':

        search = RandomizedSearchCV(regressor,
                                    param_distributions=param_space,
                                    scoring=scoring,
                                    cv=hp_cv,
                                    n_iter=n_iter,
                                    refit=False,
                                    random_state=random_state,
                                    n_jobs=4,
                                    return_train_score=True)

        ts = datetime.datetime.now()
        search.fit(x_a_train, y_a_train_true)
        log.info("dTsearch {}".format(datetime.datetime.now()-ts))

        rp.print_folds_results(search, 'et_wqa_sel{}_cv{}_it{}_search_results.csv'.format(selector.get_params()['k'], hp_cv.get_n_splits(), n_iter))
        if isinstance(scoring, str):
            log.info("Best parameters set:\n{}".format(pp.pformat(search.best_params_)))

    elif mode == 'evaluate':

        regressor.set_params(**eval_params)

        # For model evaluation / Experiments
        scores = cross_validate(regressor, x_w_train, y_w_train_true,
                                scoring=['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                                         'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2'],
                                cv=eval_cv,
                                n_jobs=4,
                                return_train_score=True)
        io.save_data(pd.DataFrame(scores), 'et_wqTwEw_sel{}_def_cv{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))

        # For model evaluation / Experiments
        scores = cross_validate(regressor, x_r_train, y_r_train_true,
                                scoring=['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                                         'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2'],
                                cv=eval_cv,
                                n_jobs=4,
                                return_train_score=True)
        io.save_data(pd.DataFrame(scores), 'et_wqTrEr_sel{}_def_cv{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))

        # For model evaluation / Experiments
        scores = cross_validate(regressor, x_a_train, y_a_train_true,
                                scoring=['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                                         'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2'],
                                cv=eval_cv,
                                n_jobs=4,
                                return_train_score=True)
        io.save_data(pd.DataFrame(scores), 'et_wqTaEa_sel{}_def_cv{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))

    elif mode == 'evaluate_rwa':

        regressor.set_params(**eval_params)

        mean_squared_error_test_r, mean_squared_error_test_w, mean_squared_error_test_a = [], [], []
        r2_test_r, r2_test_w, r2_test_a = [], [], []

        for i in range(eval_cv.get_n_splits()):
            x_a_train_split, x_a_test_split, y_a_train_true_split, y_a_test_true_split = train_test_split(x_a_train, y_a_train_true)
            # This results in a different test set as from above!!! quasi mit Zurücklegen und set filter
            _, x_r_test_split, _, y_r_test_true_split = train_test_split(x_r_train, y_r_train_true)
            _, x_w_test_split, _, y_w_test_true_split = train_test_split(x_w_train, y_w_train_true)

            regressor.fit(x_a_train_split, y_a_train_true_split)

            y_r_test_pred_split = regressor.predict(x_r_test_split)
            y_w_test_pred_split = regressor.predict(x_w_test_split)
            y_a_test_pred_split = regressor.predict(x_a_test_split)

            r2_test_r.append(r2_score(y_r_test_true_split, y_r_test_pred_split))
            r2_test_w.append(r2_score(y_w_test_true_split, y_w_test_pred_split))
            r2_test_a.append(r2_score(y_a_test_true_split, y_a_test_pred_split))

            mean_squared_error_test_r.append(mean_squared_error(y_r_test_true_split, y_r_test_pred_split))
            mean_squared_error_test_w.append(mean_squared_error(y_w_test_true_split, y_w_test_pred_split))
            mean_squared_error_test_a.append(mean_squared_error(y_a_test_true_split, y_a_test_pred_split))

        r = pd.DataFrame({'test_r2': r2_test_r, 'test_mean_squared_error': mean_squared_error_test_r})
        w = pd.DataFrame({'test_r2': r2_test_w, 'test_mean_squared_error': mean_squared_error_test_w})
        a = pd.DataFrame({'test_r2': r2_test_a, 'test_mean_squared_error': mean_squared_error_test_a})

        io.save_data(r, 'et_wqTaEr_sel{}_def_ttsr{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))
        io.save_data(w, 'et_wqTaEw_sel{}_def_ttsr{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))
        io.save_data(a, 'et_wqTaEa_sel{}_def_ttsr{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))