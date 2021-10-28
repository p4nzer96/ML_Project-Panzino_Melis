# %%

import pandas as pd
import numpy as np
import time
import sys

sys.path.insert(0, "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\")

from sklearn.metrics import accuracy_score
from KNNClassifier import KNNClassifier
from LoadDataset import LoadDataset
from utilities import time_conversion


# %%

def train_KNN_for_val():
    ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"  # Path di Andrea dei dataset

    # Creating a LoadDataset object

    dataset_loader = LoadDataset(ds_path)

    # Creating the fold list

    fold_list = [0, 1, 2, 3] * 5

    # Creation of a list that contains the values of k used

    k_max = 167
    n_folds = 4

    k_values = np.linspace(1, k_max, k_max, endpoint=True, dtype="int")

    # Creation of the vectors that will contain the results

    accuracies = np.ndarray((n_folds, k_max))
    train_times = np.ndarray((n_folds, k_max), dtype=object)
    val_times = np.ndarray((n_folds, k_max), dtype=object)
    train_times_s = np.ndarray((n_folds, k_max))
    val_times_s = np.ndarray((n_folds, k_max))

    # Starting of the process

    for fold in range(n_folds):  # ... For each fold

        train_x, train_y, val_x, val_y = dataset_loader.load_for_kfold(n_folds, fold)

        for k_cnt, k in enumerate(k_values):  # ... For each value of k

            classifier = KNNClassifier(k)  # Setting the value of k within the classifier

            # Training

            start_time = time.time()
            classifier.fit(train_x, train_y)
            h, m, s, train_times_s[fold, k_cnt] = time_conversion(time.time() - start_time)
            train_times[fold, k_cnt] = "{} h, {} m, {} s".format(h, m, s)

            # Validation

            start_time = time.time()
            y_pred, _ = classifier.predict(val_x, val_y)
            h, m, s, val_times_s[fold, k_cnt] = time_conversion(time.time() - start_time)
            val_times[fold, k_cnt] = "{} h, {} m, {} s".format(h, m, s)

            # Computing the accuracies

            accuracies[fold, k_cnt] = accuracy_score(val_y, y_pred)
            print(k)

    values = np.vstack((accuracies, train_times, train_times_s, val_times, val_times_s))

    train_df = pd.DataFrame(values, columns=k_values, index=fold_list)
    train_df.to_csv("KNN\\KNN_train_results.csv", index=False)


# %%

def train_and_testKNN(k=None):
    ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"  # Path di Andrea dei dataset

    # Creating a LoadDataset object

    dataset_loader = LoadDataset(ds_path)

    train_x, train_y = dataset_loader.load_training()
    test_x, test_y = dataset_loader.load_test()

    if k is None:

        try:

            train_df = pd.read_csv("KNN_train_results.csv")

        except FileNotFoundError:

            print("File not found!")
            return

        k = np.argmax(np.mean(train_df.values[0:4, :].astype(float), axis=0)) + 1

    classifier = KNNClassifier(k)

    # Training

    start_time = time.time()
    classifier.fit(train_x, train_y)
    h, m, s, train_time_s = time_conversion(time.time() - start_time)
    train_time = "{} h, {} m, {} s".format(h, m, s)

    # Test

    start_time = time.time()
    y_pred, conf_matrix, accuracy, fpr, fnr = classifier.predict(test_x, test_y)
    h, m, s, test_time_s = time_conversion(time.time() - start_time)
    test_time = "{} h, {} m, {} s".format(h, m, s)

    columns = ["Accuracy", "FPR", "FNR", "K Value", "Train Time", "Train Time (s)", "Test Time",
               "Test Time (s)"]

    values = [accuracy, fpr, fnr, k, train_time, train_time_s, test_time, test_time_s]

    test_df = pd.DataFrame([values], columns=columns).to_csv("KNN_test_results.csv", index=False)
    
    np.save("KNN_predictions.npy", y_pred)
    np.save("KNN_Conf_Matrix.npy", conf_matrix)


# %%

# RUNNING DELLO SCRIPT

train_and_testKNN()
