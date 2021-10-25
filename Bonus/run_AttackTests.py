import time

from secml.ml import CClassifierSVM, CClassifierKNN

from Bonus.KNN_PoisAttacker import KNN_PoisAttacker
from SVM_PoisAttacker import SVM_PoisAttacker


def run_tests(classifiers):
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

            for point in range(1, 51, 10):

                start_time = time.time()
                print(point)

                attacker.attack_classifier(100, 50, point)
                print(time.time() - start_time)

        if curr_classifier == "KNN":

            classifier = CClassifierKNN(n_neighbors=12)
            attacker = KNN_PoisAttacker(classifier)

            attacker.train_ds = tr_path
            attacker.test_ds = ts_path

            for point in range(1, 51, 10):
                start_time = time.time()
                print(point)

                attacker.attack_classifier(100, 50, point)
                print(time.time() - start_time)

        if curr_classifier == "NN":

            pass


if __name__ == "__main__":

    run_tests(["KNN", "NN"])

