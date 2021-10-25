import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\")

from utilities import compute_perf


class KNNClassifier(KNeighborsClassifier):

    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def fit(self, tr_x, tr_y):

        tr_x = np.asarray(tr_x)
        tr_y = np.asarray(tr_y)

        super().fit(tr_x, tr_y)

    def predict(self, ts_x, ts_y=None):

        ts_x = np.asarray(ts_x)

        y_pred = super().predict(ts_x)

        if ts_y is not None:

            ts_y = np.asarray(ts_y)
            cm = confusion_matrix(ts_y, y_pred)

            base_acc, fpr, fnr = compute_perf(ts_y, y_pred)

            return y_pred, cm, base_acc, fpr, fnr

        else:

            return y_pred
