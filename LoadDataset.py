import numpy as np
from secml.data import CDataset
from secml.array import CArray


# Helper class for load a CDataset and convert into ndarray

class LoadDataset:

    def __init__(self, pathname):
        self._pathname = pathname
        self._tr_x = None
        self._tr_y = None
        self._ts_x = None
        self._ts_y = None

    @property
    def tr_x(self):
        return self._tr_x

    @tr_x.setter
    def tr_x(self, value):
        self._tr_x = value

    @property
    def tr_y(self):
        return self._tr_y

    @tr_y.setter
    def tr_y(self, value):
        self._tr_y = value

    @property
    def ts_x(self):
        return self._ts_x

    @ts_x.setter
    def ts_x(self, value):
        self._ts_x = value

    @property
    def ts_y(self):
        return self._ts_y

    @ts_y.setter
    def ts_y(self, value):
        self._ts_y = value

    # Loading of training dataset

    def load_training(self):

        train_dataset = CDataset.load(self._pathname + "training_set.gz")

        self.tr_x = CArray.tondarray(train_dataset.X)
        self.tr_y = CArray.tondarray(train_dataset.Y)

        return self.tr_x, self.tr_y

    # Loading of test dataset

    def load_test(self):

        test_dataset = CDataset.load(self._pathname + "test_set.gz")

        self.ts_x = CArray.tondarray(test_dataset.X)
        self.ts_y = CArray.tondarray(test_dataset.Y)

        return self.ts_x, self.ts_y

    # Loading of full dataset (join of training and test)

    def load_all(self):

        test_dataset = CDataset.load(self._pathname + "test_set.gz")
        train_dataset = CDataset.load(self._pathname + "training_set.gz")

        self.ts_x = CArray.tondarray(test_dataset.X)
        self.ts_y = CArray.tondarray(test_dataset.Y)

        self.tr_x = CArray.tondarray(train_dataset.X)
        self.tr_y = CArray.tondarray(train_dataset.Y)

        tot_dataset = np.vstack((self._tr_x, self._ts_x))
        tot_labels = np.concatenate((self._tr_y, self._ts_y))

        return tot_dataset, tot_labels

    # Loading for k-fold cross validation

    def load_for_kfold(self, k_tot, curr_k):

        # Loading of training dataset

        self.load_training()

        # Computing the length of each fold

        split_length = int(self.tr_x.shape[0] / k_tot)

        # Creating empty arrays

        train_dataset = np.empty((0, 2000), int)
        test_dataset = np.empty((0, 2000), int)
        train_labels = np.empty(0, int)
        test_labels = np.empty(0, int)

        # Creating the folds

        for i in range(k_tot):

            if i == curr_k:

                test_dataset = np.vstack((test_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                test_labels = np.hstack((test_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

            else:

                train_dataset = np.vstack((train_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                train_labels = np.hstack((train_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

        # Reassignment of class attributes

        self.tr_x = train_dataset
        self.tr_y = train_labels
        self.ts_x = test_dataset
        self.ts_y = test_labels

        return self.tr_x, self.tr_y, self.ts_x, self.ts_y


