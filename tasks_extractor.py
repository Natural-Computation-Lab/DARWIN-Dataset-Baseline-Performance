from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
######################
# extract from the entire dataset only the features of the tasks presents in the list tasks paramenter.
# custom transformer for sklearn pipeline
class TasksExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, tasks, feature_for_task):
        self.tasks = tasks
        self.feature_for_task = feature_for_task

    def transform(self, X):
        col_list = []
        for task in self.tasks:
            for col in range((self.feature_for_task * task), self.feature_for_task * (task + 1)):
                col_list.append(X[:, col:col + 1])

        a = np.concatenate(col_list, axis=1)
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self