from copy import deepcopy
from pandas import DataFrame
import file_io as io
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import itertools


def print_folds_results(search, file_name):
    """Prints the configuration and statistics of each fold."""

    cv_results = deepcopy(search.cv_results_)

    log.debug("Detailed folds results:")
    # Remove the redundant params list
    cv_results.pop('params', None)

    df_results = DataFrame(cv_results)
    log.debug("\n{}".format(df_results))

    io.store_search_data(df_results, file_name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        log.info("Normalized confusion matrix")
    else:
        log.info('Confusion matrix, without normalization')

    log.info('\n{}'.format(cm))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
