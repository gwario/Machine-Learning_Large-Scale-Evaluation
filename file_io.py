import logging as log
import os.path
import json
import numpy as np
import arff
from pprint import pprint, pformat
import pandas as pd
from sklearn.externals import joblib

__doc__ = """
Contains code to load and save data.
"""
__author__ = "Mario Gastegger"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "2.0"
__status__ = "Development"


bs_features = ['id', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp',
               'atemp', 'hum', 'windspeed']
bs_id = 'id'
bs_target = 'cnt'
# output attrs: id, cnt

wq_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
wq_target = 'quality'

sp_features = ['id', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
               'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
               'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
sp_id = 'id'
sp_target = 'Grade'
# output attrs: id, Grade

ch_target = 'prp'
ch_features = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax']

# use at most DATASET_INSTANCE_CAP_PCT of the instances
DATASET_LESS_DATA = True
DATASET_INSTANCE_CAP_CNT = 1000


def load_data(data_file, config):
    """Loads the input data from data_file."""

    log.debug("Loading data...")

    if data_file == 'bikeSharing.shuf.train.csv':
        tp = pd.read_csv(data_file, iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)

        df = df.drop([bs_id], axis=1)

        x_train = df.drop(bs_target, axis=1)
        y_train = df.loc[:, bs_target].values.ravel()

        return x_train, y_train

    elif data_file == 'bikeSharing.shuf.test.csv':
        tp = pd.read_csv(data_file, iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)
        y_test = df[bs_id]
        x_test = df.drop([bs_id], axis=1)

        return x_test, y_test

    elif data_file.startswith('winequality-'):
        tp = pd.read_csv(data_file, sep=';', iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)

        x_train = df.drop(wq_target, axis=1)
        y_train = df.loc[:, wq_target].values.ravel()

        return x_train, y_train

    elif data_file == 'StudentPerformance.shuf.train.csv':
        tp = pd.read_csv(data_file, iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)

        df = df.drop([sp_id], axis=1)

        x_train = df.drop(sp_target, axis=1)
        y_train = df.loc[:, sp_target].values.ravel()

        return x_train, y_train

    elif data_file == 'StudentPerformance.shuf.test.csv':
        tp = pd.read_csv(data_file, iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)
        y_test = df[sp_id]
        x_test = df.drop([sp_id], axis=1)

        return x_test, y_test

    elif data_file == 'machine.data.csv':

        tp = pd.read_csv(data_file, iterator=True, chunksize=1000, verbose=True)
        df = pd.concat(tp, ignore_index=True)
        df = df[ch_features+[ch_target]]

        x_train = df.drop(ch_target, axis=1)
        y_train = df.loc[:, ch_target].values.ravel()

        return x_train, y_train

    elif data_file == 'iris.arff':

        #data, meta = arff.loadarff(data_file)
        data = arff.load(open(data_file))
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])
        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
        y = df.loc[:, ['class']].values.ravel()
        #y = [element.decode('utf-8') for element in y]
        #y = df.loc[:, ['class']]

        return X, y, data

    elif data_file == 'mammography.arff':

        #data, meta = arff.loadarff(data_file)
        data = arff.load(open(data_file))
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])
        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6']]
        y = df.loc[:, ['class']].values.ravel()
        #y = [element.decode('utf-8') for element in y]

        return X, y, data

    elif data_file == 'speeddating.arff':

        data = arff.load(open(data_file), encode_nominal=True)
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])

        df.info()

        # nan: age, like, guess_prob_liked, met
        log.warning('Replacing NaN/missing values with 0...')
        df.fillna(0, inplace=True)

        #log.warning('Dropping instances with NaN/missing values...')
        #before = df.shape[0]
        #df.dropna(axis=0, how='any', inplace=True)
        #after = df.shape[0]
        #log.warning('Dropped {} instances of {}!'.format(before-after, before))

        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, [name for (name, type) in data['attributes'] if name != 'match']]
        y = df.loc[:, ['match']].values.ravel()

        return X, y, data

    elif data_file == 'climate-model-simulation-crashes.arff':

        data = arff.load(open(data_file))
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])

        df.info()

        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, [name for (name, type) in data['attributes'] if name != 'Class']]
        y = df.loc[:, ['Class']].values.ravel()

        return X, y, data

    elif data_file == 'diabetes.arff':

        data = arff.load(open(data_file))
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])

        df.info()

        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, [name for (name, type) in data['attributes'] if name != 'class']]
        y = df.loc[:, ['class']].values.ravel()

        return X, y, data

    elif data_file == 'ilpd.arff':

        data = arff.load(open(data_file), encode_nominal=True)
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])

        df.info()

        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, [name for (name, type) in data['attributes'] if name != 'Class']]
        y = df.loc[:, ['Class']].values.ravel()

        return X, y, data

    elif data_file == 'kc2.arff':

        data = arff.load(open(data_file))
        df = pd.DataFrame(data['data'], columns=[name for (name, type) in data['attributes']])
        df = df.sample(frac=1, random_state=config['random_state'])

        df.info()

        if DATASET_LESS_DATA:
            log.warning("Using at most {} instances.".format(DATASET_INSTANCE_CAP_CNT))
            df.drop(df.index[:-DATASET_INSTANCE_CAP_CNT], inplace=True)

        df.info()

        X = df.loc[:, [name for (name, type) in data['attributes'] if name != 'problems']]
        y = df.loc[:, ['problems']].values.ravel()

        return X, y, data


def load_config(argv, default_config):

    if len(argv) > 1:
        with open(argv[1], 'r') as config_file:
            log.info("Using configuration from: {}".format(config_file.name))
            return json.load(config_file)
    else:
        log.warning("Using default config.")
        return default_config


def load_model(model_file):
    """Loads and returns the pipeline model from model_file."""

    log.debug("Loading model from {}".format(model_file))
    model = joblib.load(model_file)
    return model[0], model[1]


def save_model(fu_pl, clf_pl, model_filename):
    """Saves the pipeline model to model_filename."""

    joblib.dump([fu_pl, clf_pl], model_filename)
    print("Model saved as {}".format(model_filename))


def save_data(dataset, dataset_filename):
    """Saves the dataset to dataset_filename."""

    dataset.to_csv(dataset_filename, sep=',', index=False, encoding='utf-8')
    print("Dataset saved as {}".format(dataset_filename))


def save_data_arff(dataset, dataset_filename, arff_data):
    """Saves the dataset, arff-formatted, to dataset_filename."""
    #pprint(arff_data)
    #print(dataset)
    #print(type(arff_data['data']))
    arff_data['data'] = dataset.values.tolist()
    #print(type(arff_data['data']))
    arff.dump(arff_data, open(dataset_filename, 'w'))
    print("Dataset saved as {}".format(dataset_filename))


def save_prediction(prediction, prediction_filename):
    """Saves the prediction to prediction_filename."""

    prediction.to_csv(prediction_filename, sep=',', index=False, encoding='utf-8')
    print("Prediction saved as {}".format(prediction_filename))


def load_additional_data(data_file):
    """Loads the keywords data from <data_file>.add."""

    log.debug("Loading data...")
    tp = pd.read_csv(data_file + ".add", iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)

    return df


def save_additional_data(data, filename):
    """Saves the data to <filename>.add."""
    data.to_csv(filename + ".add", sep=',', index=False, encoding='utf-8')
    print("Additional data saved as {}".format(filename))


def store_oob_score_data(params, oob_errors):
    """Stores the oob error data in to a oob.csv"""
    filename = "oob.csv"
    data = pd.DataFrame(oob_errors)
    #print(data)

    data.to_csv(filename, sep=',', index=False, encoding='utf-8')
    print("OOB Score saved as {}".format(filename))


def store_search_data(data, file_name):
    """Stores the oob error data in to a oob.csv"""

    data.to_csv(file_name, sep=',', index=False, encoding='utf-8')
    print("Search results saved as {}".format(file_name))


def store_hyper_parameters(parameters, filename, working_dir):
    """Stores the hyper-parameter."""

    with open(os.path.join(filename, working_dir), 'w') as outfile:
        json.dump(parameters, outfile)

    print("Saved hyper-parameters as {}".format(os.path.join(filename, working_dir)))


def save_config(config, file_name):
    """Stores the actual configuration."""

    with open(file_name, 'w') as outfile:
        json.dump(config, outfile)