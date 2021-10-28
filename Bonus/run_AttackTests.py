import time

import numpy as np
import pandas as pd
from secml.ml import CClassifierSVM

import utilities
from SVM_PoisAttacker import SVM_PoisAttacker


def run_tests(classifiers):
    # Training and Test dataset paths

    tr_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\training_set.gz"
    ts_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\test_set.gz"

    for curr_classifier in classifiers:

        if curr_classifier == "SVM":

            # I create the classifier by choosing, as hyperparameters, those that gave the best results obtained in
            # the previous experiments

            classifier = CClassifierSVM(kernel="rbf")
            attacker = SVM_PoisAttacker(classifier)

            attacker.train_ds = tr_path
            attacker.test_ds = ts_path

            #attacker.train_ds, attacker.test_ds = utilities.filter_ds()

            diffs = []
            pois_accs = []
            no_pois_accs = []
            times = []
            cnt = 1

            min_pts = 20
            max_pts = 100
            step_pts = 20

            for point in range(min_pts, max_pts + step_pts, step_pts):

                start_time = time.time()
                print(point)
                pois_acc, no_pois_acc = attacker.attack_classifier(1000, 2000, point)
                total_time = time.time() - start_time

                pois_accs.append(pois_acc)
                no_pois_accs.append(no_pois_acc)
                times.append(total_time)
                diffs.append(no_pois_acc - pois_acc)

                if pois_acc is None or no_pois_acc is None or point == max_pts:

                    values = np.zeros(shape=(cnt, 4))

                    values[:, 0] = np.asarray(no_pois_accs)
                    values[:, 1] = np.asarray(pois_accs)
                    values[:, 2] = np.asarray(diffs)
                    values[:, 3] = np.asarray(times)

                    columns = ["No Pois. Acc.", "Pois. Acc", "Difference", "Times"]

                    values_df = pd.DataFrame(values, columns=columns)

                    values_df.to_csv("SVM_pois_results.csv")

                    break

                else:

                    cnt += 1

        if curr_classifier == "KNN":
            pass

        if curr_classifier == "NN":
            pass


if __name__ == "__main__":
    run_tests(["SVM"])
