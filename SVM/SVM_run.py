#%%
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\")

from utilities import time_conversion
from LoadDataset import LoadDataset
from SVMClassifier import SVMClassifier



#%%

ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"  # Path di Andrea dei dataset
dataset_loader = LoadDataset(ds_path)

kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']

# Imposto le liste che conterranno i risultati di ogni fold

accuracies = []

# Durata del test e validation in ore, minuti, secondi

train_times = []
test_times = []

# Durata del test e validation in secondi

train_total_times = []
test_total_times = []

# Lista dei Kernel e dei Fold utilizzati

kernels_used = []
folds = []

#%%

# Avvio dell'algoritmo di training e validation

for kernel in kernel_list:

    print("Kernel used: {}".format(kernel))
    classifier = SVMClassifier(kernel=kernel)

    for fold in range(4):

        train_x, train_y, val_x, val_y = dataset_loader.load_for_kfold(4, fold)

        start_time = time.time()
        classifier.fit(train_x, train_y)  # Train del classificatore
        end_time = time.time()
        h, m, s, train_time = time_conversion(end_time - start_time)

        train_times.append("{} h, {} m, {} s".format(h, m, s))

        start_time = time.time()
        acc = classifier.predict(val_x, val_y)  # Test del classificatore
        end_time = time.time()
        h, m, s, test_time = time_conversion(end_time - start_time)

        test_times.append("{} h, {} m, {} s".format(h, m, s))

        kernels_used.append(kernel)
        folds.append(fold)
        accuracies.append(round(acc*100, 2))
        train_total_times.append(train_time)
        test_total_times.append(test_time)


columns = ["Kernel Used", "Fold N°", "Validation Accuracy", "Train Time", "Train Time (s)", "Validation Time",
           "Validation Time (s)"]

values = np.array([kernels_used, folds, accuracies, train_times, train_total_times, test_times, test_total_times]).T

train_df = pd.DataFrame(values, columns=columns, index=None)
train_df.to_csv("SVM\\SVM_train_results.csv", index=False)

# %%

train_x, train_y = dataset_loader.load_training()
test_x, test_y = dataset_loader.load_test()

classifier = SVMClassifier(kernel="rbf")

start_time = time.time()
classifier.fit(train_x, train_y)  # Train del classificatore
end_time = time.time()
h, m, s, train_total_time = time_conversion(end_time - start_time)

train_time = "{} h, {} m, {} s".format(h, m, s)

start_time = time.time()
y_pred, conf_matrix, accuracy, fpr, fnr  = classifier.predict(test_x, test_y)  # Test del classificatore
end_time = time.time()
h, m, s, test_total_time = time_conversion(end_time - start_time)

test_time = "{} h, {} m, {} s".format(h, m, s)

columns = ["Kernel Used", "Test Accuracy", "FPR", "FNR", "Train Time", "Train Time (s)", "Test Time",
           "Test Time (s)"]

values = np.array(["rbf", accuracy, fpr, fnr, train_time, train_total_time, test_time, test_total_time])
test_df = pd.DataFrame([values], columns=columns, index=None)
test_df.to_csv("SVM_test_results.csv", index=False)

np.save("SVM_predictions.npy", y_pred)
np.save("SVM_Conf_Matrix.npy", conf_matrix)
# %%
