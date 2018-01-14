#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function, absolute_import

import logging as log
import pprint as pp
import os
import sys
import json
import numpy as np
import datetime

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate, KFold, train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_regression

import file_io as io
import report as rp

# Display progress logs on stdout
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Template for config.json
default_config = {
    'experiment': 'default_experiment',         # the title of the experiment
    'stratify': True,                           # whether to stratify or not
    'repetitions': 3,                           # number of times every estimator is run with every dataset
    'datasets': ['speeddating.arff',            # preprocessed dataset
                 'mammography.arff',
                 'iris.arff',],
    'random_state': 12345,                      # the random state used where possible
    'train_size': 0.6,                          # size of training set
    'test_splits': 2,                           # the number of test splits (test_size = (1-train_size)/test_spits)
    'estimators': [                             # the estimators
        {
            'estimator': 'ExtraTreesClassifier',# estimator name from the list of available estimators
            'params': {                         # estimator parameters, see scikit docs
                'random_state': 12345,          # estimators don't use the global random_state, set it for reproducibility
                'n_estimators': 500
            }
        },
        {
            'estimator': 'RandomForestClassifier',
            'params': {
                'random_state': 12345,
                'n_estimators': 500}
        },
    ]
}


def generate_test_sets(X_test, y_test, config):

    log.debug("# TEST_SETS: {}".format(config['test_splits']))
    log.debug("TEST_SIZE: {}".format((1 - config['train_size']) / config['test_splits']))

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


def get_estimator(config):

    estimator_name = config['estimator']
    estimator_params = config['params']

    if estimator_name == 'RandomForestClassifier':
        estimator = RandomForestClassifier()
    elif estimator_name == 'ExtraTreesClassifier':
        estimator = ExtraTreesClassifier()
    elif estimator_name == 'AdaBoostClassifier':
        estimator = AdaBoostClassifier()
    elif estimator_name == 'DecisionTreeClassifier':
        estimator = DecisionTreeClassifier()
    else:
        raise ValueError("Unknown estimator '{}'".format(estimator_name))

    estimator.set_params(**estimator_params)
    config['params'] = estimator.get_params()

    return estimator


# Available datasets:
#
# Available estimators:
# RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, AdaBoostClassifier ...
# TODO add more estimators
#


if __name__ == '__main__':

    config = io.load_config(sys.argv, default_config)

    experiment_dir = config['experiment']
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    times = pd.DataFrame(columns=['dataset_name', 'estimator_name', 'repetition', 'split', 'fit_time', 'pred_time'])
    scores = pd.DataFrame(columns=['dataset_name', 'estimator_name', 'repetition', 'split', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])

    for dataset_i, dataset_filename in enumerate(config['datasets']):
        log.debug("DATASET_{}: {}".format(dataset_i, dataset_filename))

        # load preprocessed dataset
        X, y, arff_data = io.load_data(dataset_filename, config)
        dataset_name = os.path.splitext(dataset_filename)[0]
        log.info("DATASET_{}_NAME: {}".format(dataset_i, dataset_name))

        for estimator_i, estimator in enumerate(config['estimators']):
            log.debug("ESTIMATOR_{}: {}".format(estimator_i, estimator['estimator']))

            # load algorithm
            estimator = get_estimator(estimator)
            estimator_name = estimator.__class__.__name__
            log.info("ESTIMATOR_{}_NAME: {}".format(estimator_i, estimator_name))

            # for every repetition
            for repetition_i in range(config['repetitions']):
                log.info("REPETITION_{}".format(repetition_i))

                X_train, X_test, y_train_true, y_test_true = train_test_split(X, y,
                                                                    stratify=y if config['stratify'] else None,
                                                                    train_size=config['train_size'],
                                                                    random_state=config['random_state']+repetition_i)
                log.debug("TRAIN_SET:\n{}".format(pp.pformat(X_train)))
                log.debug("WHOLE_TEST_SET:\n{}".format(pp.pformat(X_test)))

                # generate test sets
                X_test_sets, y_test_sets = generate_test_sets(X_test, y_test_true, config)
                log.debug("TEST_SETS:\n{}".format(pp.pformat(X_test_sets)))

                t0 = datetime.datetime.now()
                estimator.fit(X_train, y_train_true)
                dT_fit = datetime.datetime.now() - t0

                t0 = datetime.datetime.now()
                y_train_pred = estimator.predict(X_train)
                dT_train_pred = datetime.datetime.now() - t0

                # store predicted values in arff:
                # d<dataset_id>_<dataset_name>_e<estimator_id>_<estimator_name>_r<repetition>_<set>[set_id].arff
                X_train.loc[:, 'class'] = pd.Series(y_train_pred, index=X_train.index)
                io.save_data_arff(X_train,
                                  '{}/d{}_{}_e{}_{}_r{}_train.arff'.format(experiment_dir,
                                                                           dataset_i, dataset_name,
                                                                           estimator_i, estimator_name,
                                                                           repetition_i),
                                  arff_data)

                train_accuracy = accuracy_score(y_train_true, y_train_pred)
                train_f1 =  f1_score(y_train_true, y_train_pred, average='macro')
                train_precision = precision_score(y_train_true, y_train_pred, average='macro')
                train_recall = recall_score(y_train_true, y_train_pred, average='macro')

                log.info("Classification report for train:\n{}".format(classification_report(y_train_true, y_train_pred)))

                row = pd.Series(["d{}_{}".format(dataset_i, dataset_name),
                                 "e{}_{}".format(estimator_i, estimator_name),
                                 "r{}".format(repetition_i),
                                 "train",
                                 dT_fit, dT_train_pred],
                                index=['dataset_name', 'estimator_name', 'repetition', 'split', 'fit_time', 'pred_time'])
                times = times.append(row, ignore_index=True)

                row = pd.Series(["d{}_{}".format(dataset_i, dataset_name),
                                 "e{}_{}".format(estimator_i, estimator_name),
                                 "r{}".format(repetition_i),
                                 "train",
                                 train_accuracy, train_f1, train_precision, train_recall],
                                index=['dataset_name', 'estimator_name', 'repetition', 'split',
                                       'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
                scores = scores.append(row, ignore_index=True)

                # evaluation with all metrics for X_train and all in X_test_sets
                for test_set_i in range(len(X_test_sets)):

                    X_test = X_test_sets[test_set_i]
                    y_test_true = y_test_sets[test_set_i]

                    t0 = datetime.datetime.now()
                    y_test_pred = estimator.predict(X_test)
                    dT_train_pred = datetime.datetime.now() - t0

                    X_test.loc[:, 'class'] = pd.Series(y_test_pred, index=X_test.index)
                    io.save_data_arff(X_test,
                                      '{}/d{}_{}_e{}_{}_r{}_test{}.arff'.format(experiment_dir,
                                                                                dataset_i, dataset_name,
                                                                                estimator_i, estimator_name,
                                                                                repetition_i, test_set_i),
                                      arff_data)

                    test_accuracy = accuracy_score(y_test_true, y_test_pred)
                    test_f1 =  f1_score(y_test_true, y_test_pred, average='macro')
                    test_precision = precision_score(y_test_true, y_test_pred, average='macro')
                    test_recall = recall_score(y_test_true, y_test_pred, average='macro')

                    log.info("Classification report for test_{}:\n{}".format(test_set_i, classification_report(y_test_true, y_test_pred)))

                    row = pd.Series(["d{}_{}".format(dataset_i, dataset_name),
                                     "e{}_{}".format(estimator_i, estimator_name),
                                     "r{}".format(repetition_i),
                                     "test{}".format(test_set_i),
                                     dT_fit, dT_train_pred],
                                    index=['dataset_name', 'estimator_name', 'repetition', 'split',
                                           'fit_time', 'pred_time'])
                    times = times.append(row, ignore_index=True)

                    row = pd.Series(["d{}_{}".format(dataset_i, dataset_name),
                                     "e{}_{}".format(estimator_i, estimator_name),
                                     "r{}".format(repetition_i),
                                     "test{}".format(test_set_i),
                                     test_accuracy, test_f1, test_precision, test_recall],
                                    index=['dataset_name', 'estimator_name', 'repetition', 'split',
                                           'accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
                    scores = scores.append(row, ignore_index=True)


    #pp.pprint(scores)
    #pp.pprint(times)

    # store metrics
    io.save_config(config, experiment_dir+'/config.json')
    io.save_data(scores, experiment_dir+'/evaluation_scores.csv')
    io.save_data(times, experiment_dir+'/evaluation_times.csv')

    # TODO calculate mean and stdev among repetitions

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


        io.save_data(w, 'et_wqTaEw_sel{}_def_ttsr{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))
        io.save_data(a, 'et_wqTaEa_sel{}_def_ttsr{}_evaluation_scores.csv'.format(selector.get_params()['k'], eval_cv.get_n_splits()))