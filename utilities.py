import numpy as np
from secml.data import CDataset
from sklearn.metrics import accuracy_score

from LoadDataset import LoadDataset


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

    fpr = round(fp / (fp + tn) * 100, 2)
    fnr = round(fn / (fn + tp) * 100, 2)
    acc = round(acc, 2)

    return acc, fpr, fnr


def filter_ds(pad=True):
    index = 0
    index_list = []

    dataset_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"

    ds_loader = LoadDataset(dataset_path)

    train_x, train_y = ds_loader.load_training()
    val_x, val_y = ds_loader.load_test()

    with open(dataset_path + "features.txt") as f:

        line_count = 0
        for line in f:
            if line != "\n" and "req_perm" in line:
                line_count += 1
                index_list.append(index)
            index += 1
        f.close()

    # Filtro il dataset

    if pad:

        f_train_x = np.zeros(shape=train_x.shape)
        f_val_x = np.zeros(shape=val_x.shape)

        for item in index_list:
            # Sto prendendo solo i valori presenti a determinati indici

            f_train_x[:, item] = train_x[:, item]
            f_val_x[:, item] = val_x[:, item]

    else:

        f_train_x = np.zeros(shape=(train_x.shape[0], len(index_list)))
        f_val_x = np.zeros(shape=(val_x.shape[0], len(index_list)))

        for idx, item in enumerate(index_list):
            f_train_x[:, idx] = train_x[:, item]
            f_val_x[:, idx] = val_x[:, item]

            print(f_train_x.shape[1])

    f_train_ds = CDataset(f_train_x, train_y)
    f_val_ds = CDataset(f_val_x, val_y)

    return f_train_ds, f_val_ds
