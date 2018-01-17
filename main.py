#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function, absolute_import

import datetime
import logging as log
import os
import pprint as pp
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

import file_io as io
import util as ut

# Display progress logs on stdout
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')


# Template for config.json
default_config = {
    'experiment': 'default_experiment',         # the title of the experiment
    'stratify': True,                           # whether to stratify or not
    'repetitions': 2,                           # number of times every estimator is run with every dataset
    'datasets': ['iris.arff',                   # preprocessed dataset
                 'mammography.arff',],
    'random_state': 12345,                      # the random state used where possible
    'train_size': 0.6,                          # size of training set
    'test_splits': 2,                           # the number of test splits (test_size = (1-train_size)/test_spits)
    'estimators': [                             # the estimators
        {
            'estimator': 'ExtraTreesClassifier',# estimator name from the list of available estimators
            'params': {                         # estimator parameters, see scikit docs, only non objects are allowed in the json file
                'random_state': 12345,          # estimators don't use the global random_state, set it for reproducibility
                'n_estimators': 100
            }
        },
        {
            'estimator': 'RandomForestClassifier',
            'params': {
                'random_state': 12345,
                'n_estimators': 100}
        },
    ]
}

if __name__ == '__main__':
    """
    Available datasets:
     * speeddating.arff
     * mammography.arff
     * iris.arff
     * climate-model-simulation-crashes.arff
     * diabetes.arff
     * ilpd.arff
     * kc2.arff
     * steel-plates-fault.arff
     * segment.arff
     * hepatitis.arff
     * ringnorm.arff
     * credit-g.arff
     * ...
    
     Available estimators:
     * RandomForestClassifier
     * ExtraTreesClassifier
     * DecisionTreeClassifier
     * AdaBoostClassifier
     * GaussianNB
     * KNeighborsClassifier
     * SVC
     * GaussianProcessClassifier
     * MLPClassifier
     * Perceptron
     * ...
     TODO add more estimators
    """
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

        for estimator_i, estimator_obj in enumerate(config['estimators']):
            log.debug("ESTIMATOR_{}: {}".format(estimator_i, estimator_obj['estimator']))

            # load algorithm
            estimator = ut.get_estimator(estimator_obj)
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
                X_test_sets, y_test_sets = ut.generate_test_sets(X_test, y_test_true, repetition_i, config)
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
    io.save_data_arff(scores, experiment_dir+'/evaluation_scores.arff')
    io.save_data(scores, experiment_dir+'/evaluation_scores.csv')
    io.save_data_arff(times, experiment_dir+'/evaluation_times.arff')
    io.save_data(times, experiment_dir+'/evaluation_times.csv')

    # TODO save two arffs per dataset and classifier:
    # 1) results (e.g. tested accuracy, precision, TP rate, ...)
    # 2) meta-data (name and parameters of classifiers, runtime (total, and each of training and testing),
    # date of experiment, input data used (e.g. filename))

    # TODO calculate mean and stdev among repetitions (?)
    # TODO store results diagrams
    # TODO arff output template (?)
    # dataset, algorithm, repetition_index, split_index, metric1, metric2, metric3

    # TODO generate diagrams (maybe external script?)
