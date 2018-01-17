#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function, absolute_import

import logging as log
import os
import sys
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV

import file_io as io
import report as rp
import util as ut

# Display progress logs on stdout
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    """Does hp search and stores the parameters for each dataset and classifier."""

    config = io.load_config(sys.argv, None)

    experiment_dir = "{}_hpsearch".format(config['experiment'])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    for dataset_i, dataset_filename in enumerate(config['datasets']):
        log.debug("DATASET_{}: {}".format(dataset_i, dataset_filename))

        # load preprocessed dataset
        X, y, arff_data = io.load_data(dataset_filename, config)
        dataset_name = os.path.splitext(dataset_filename)[0]
        log.info("DATASET_{}_NAME: {}".format(dataset_i, dataset_name))

        for estimator_i, estimator in enumerate(config['estimators']):
            log.debug("ESTIMATOR_{}: {}".format(estimator_i, estimator['estimator']))

            # load algorithm
            estimator = ut.get_estimator(estimator)
            estimator_name = estimator.__class__.__name__
            log.info("ESTIMATOR_{}_NAME: {}".format(estimator_i, estimator_name))

            search_space, n_iter = ut.get_search_space(estimator)

            if search_space is not None:
                random_search = RandomizedSearchCV(estimator,
                                                   scoring='accuracy',
                                                   param_distributions=search_space,
                                                   cv=5,
                                                   n_iter=n_iter,
                                                   random_state=config['random_state'],
                                                   return_train_score=True)

                # Do cross validated hyper-parameter search
                t0 = datetime.now()
                random_search.fit(X, y)
                dt_search = datetime.now() - t0

                rp.print_folds_results(random_search, '{}/{}_{}.csv'.format(experiment_dir, dataset_name, estimator_name))
                log.info("Parameter search done in {}".format(dt_search))
                #log.info("Best parameters set:\n{}".format(pformat(random_search.best_params_)))
            else:
                log.info("No parameter search necessary for {}".format(estimator_name))

