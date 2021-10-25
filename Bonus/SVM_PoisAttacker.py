# Import of the libraries used

import sys

from numpy.linalg import LinAlgError
from secml.data import CDataset

sys.path.insert(0, "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\")

from secml.ml.classifiers import CClassifierSVM
from secml.data.splitter import CTrainTestSplit
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks import CAttackPoisoningSVM


# Dataset loading

class SVM_PoisAttacker:

    def __init__(self, classifier, train_ds=None, test_ds=None):
        self._classifier = classifier
        self._train_ds = train_ds
        self._test_ds = test_ds

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    @property
    def train_ds(self):
        return self._train_ds

    @train_ds.setter
    def train_ds(self, value):

        if isinstance(value, str):

            self._train_ds = CDataset.load(value)

        elif isinstance(value, CDataset):

            self._train_ds = value

        else:

            print("Error: Possible parameters accepted are the path to the dataset or an object of type CDataset")

        # Removing the header from CDatasets (their presence was causing problems in the code)

        self._train_ds.header = None

    @property
    def test_ds(self):
        return self._test_ds

    @test_ds.setter
    def test_ds(self, value):
        if isinstance(value, str):

            self._test_ds = CDataset.load(value)

        elif isinstance(value, CDataset):

            self._test_ds = value

        else:

            print("Error: Possible parameters accepted are the path to the dataset or an object of type CDataset")

        # Removing the header from CDatasets (their presence was causing problems in the code)

        self._test_ds.header = None

    def attack_classifier(self, tr_size, ts_size, pois_pts=20, test_no_pois=True):

        # I create a split of the dataset for training and validation

        splitter = CTrainTestSplit(train_size=tr_size, test_size=ts_size, random_state=123)
        train, val = splitter.split(self.train_ds)

        assert (isinstance(self.classifier, CClassifierSVM))

        metric = CMetricAccuracy()

        if test_no_pois:
            # We can now fit the classifier
            self.classifier.fit(train.X, train.Y)
            print("Training of classifier complete!")

            # Compute predictions on a test set
            y_pred = self.classifier.predict(self.test_ds.X)

            # Evaluate the accuracy of the original classifier
            no_pois_acc = metric.performance_score(y_true=self.test_ds.Y, y_pred=y_pred)
            print("Accuracy before attack on test set: {:.2%}".format(no_pois_acc))

        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'eta_max': None,
            'max_iter': 1000,
            'eps': 1e-6
        }

        # Avviamo l'attacco di poisoning del classificatore SVM

        pois_attack = CAttackPoisoningSVM(classifier=self.classifier,
                                          training_data=train,
                                          val=val,
                                          solver_params=solver_params,
                                          random_seed=21)

        # pois_attack.verbose = 2

        # chose and set the initial poisoning sample features and label
        xc = train[0, :].X
        yc = train[0, :].Y
        pois_attack.x0 = xc
        pois_attack.xc = xc
        pois_attack.yc = yc

        # Number of poisoning points to generate
        pois_attack.n_points = pois_pts

        # Run the poisoning attack
        print("Attack started...")

        try:

            pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(self.test_ds.X, self.test_ds.Y)

        except LinAlgError:

            print("Error: SVD did not converge")

            return None

        print("Attack complete!")

        # Evaluate the accuracy after the poisoning attack
        pois_acc = metric.performance_score(y_true=self.test_ds.Y, y_pred=pois_y_pred)

        print("Accuracy after attack on test set: {:.2%}".format(pois_acc))

        if test_no_pois:

            return pois_acc, no_pois_acc

        else:

            return pois_acc
