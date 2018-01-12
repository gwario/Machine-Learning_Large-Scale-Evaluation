from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class StrDateToUnixTs(BaseEstimator, TransformerMixin):
    """Prints the input."""

    def __init__(self,label = None):
        self.label = label

    def fit(self, x, y=None):
        return self

    def transform(self, input):

        input['dteday'] = pd.to_numeric(pd.to_datetime(input['dteday']))

        return input