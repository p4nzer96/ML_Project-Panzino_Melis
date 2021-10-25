import numpy as np
from sklearn.metrics import accuracy_score


def time_conversion(time):
    if isinstance(time, object):
        time = float(time)

    hours = int(time / 3600)
    minutes = int((time / 3600 - hours) * 60)
    seconds = int(time - hours * 3600 - minutes * 60)

    return hours, minutes, seconds, round(time, 2)


def compute_perf(y_pred, y_true):
    acc = accuracy_score(y_pred, y_true) * 100

    fp = 0  # False Positives
    fn = 0  # False Negatives
    tp = 0  # True Positives
    tn = 0  # True Negatives

    assert (len(y_true) == len(y_pred))

    for i in range(len(y_true)):

        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1

        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

        elif y_true[i] == 1 and y_pred[i] == 1:
            tp += 1

        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    fpr = round(fp / (fp + tn)*100, 2)
    fnr = round(fn / (fn + tp)*100, 2)
    acc = round(acc, 2)

    return acc, fpr, fnr
