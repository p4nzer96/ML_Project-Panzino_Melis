# %%

from NNClassifier import NNClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import sys

sys.path.insert(0, "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\")

from utilities import time_conversion, compute_perf
from LoadDataset import LoadDataset


# %%

def train_NN():

    # Dataset path

    ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"

    dataset_loader = LoadDataset(ds_path)

    fold_list = [0, 1, 2, 3]

    n_folds = len(fold_list)  # Number of folds
    n_epochs = 100  # Number of epochs used in the NN Classifier

    # Creation of the vectors that will contain the results

    tr_accuracies = np.ndarray((n_folds, n_epochs))
    val_accuracies = np.ndarray((n_folds, n_epochs))
    tr_losses = np.ndarray((n_folds, n_epochs))
    val_losses = np.ndarray((n_folds, n_epochs))

    train_times = np.zeros(shape=n_folds, dtype=object)
    train_times_s = np.zeros(shape=n_folds)

    classifier = NNClassifier()

    # Starting of the process

    for fold in range(n_folds):  # For every fold

        train_x, train_y, val_x, val_y = dataset_loader.load_for_kfold(
            n_folds, fold)

        start_time = time.time()
        tr_accuracies[fold, :], tr_losses[fold, :], val_accuracies[fold, :], val_losses[fold, :] = classifier.fit(
            train_x, train_y, val_x, val_y, n_tr_epochs=n_epochs, model_name="NN_Model_v3_f{}".format(fold))
        h, m, s, train_times_s[fold] = time_conversion(
            time.time() - start_time)
        train_times[fold] = "{} h, {} m, {} s".format(h, m, s)

    # Storing the results into pandas DataFrames

    train_acc_df = pd.DataFrame(tr_accuracies, columns=[
        i for i in range(1, n_epochs + 1)], index=fold_list)

    train_losses_df = pd.DataFrame(
        tr_losses, columns=[i for i in range(1, n_epochs + 1)], index=fold_list)

    val_acc_df = pd.DataFrame(val_accuracies, columns=[
        i for i in range(1, n_epochs + 1)], index=fold_list)

    val_losses_df = pd.DataFrame(
        val_losses, columns=[i for i in range(1, n_epochs + 1)], index=fold_list)

    train_times_df = pd.DataFrame(np.array([train_times, train_times_s]).T, index=fold_list)

    # Saving the dataframes to .csv

    train_acc_df.to_csv("train_accuracies.csv")
    train_losses_df.to_csv("train_losses.csv")
    val_acc_df.to_csv("val_accuracies.csv")
    val_losses_df.to_csv("val_losses.csv")

    train_times_df.to_csv("train_times.csv")


# %%

def train_and_testNN():
    ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"

    # Loading of datasets

    dataset_loader = LoadDataset(ds_path)

    train_x, val_x, train_y, val_y = train_test_split(dataset_loader.load_training()[0],
                                                      dataset_loader.load_training()[1], test_size=0.1)
    test_x, test_y = dataset_loader.load_test()

    classifier = NNClassifier()

    n_epochs = 100

    classifier.fit(train_x, train_y, val_x, val_y, n_tr_epochs=n_epochs, model_name="NN_Model_v3")

    start_time = time.time()
    y_pred = classifier.predict(test_x, batch_size=4337, model_name="NN_Model_v3.h5")
    h, m, s, test_time_s = time_conversion(time.time() - start_time)

    acc, fpr, fnr = compute_perf(y_pred, test_y)

    print("Accuracy: {} %\nFalse Positive Ratio (FPR) [Goodwares misclassified as Malwares]: {} %\nFalse Negative "
          "Ratio (FNR) [Malwares misclassified as Goodwares] {} % ".format(acc, fpr, fnr))
    print("Test Time: {}".format(test_time_s))


# %%

# RUNNING OF THE SCRIPT

train_NN()
train_and_testNN()
