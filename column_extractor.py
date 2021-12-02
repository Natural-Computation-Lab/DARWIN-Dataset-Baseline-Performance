from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
######################
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        a = np.concatenate(col_list, axis=1)
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self