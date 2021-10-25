import numpy as np
from secml.data import CDataset
from secml.array import CArray


# Classe utilizzata per il caricamento del dataset come ndarray

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

    # Caricamento del dataset di training

    def load_training(self):

        train_dataset = CDataset.load(self._pathname + "training_set.gz")

        self.tr_x = CArray.tondarray(train_dataset.X)
        self.tr_y = CArray.tondarray(train_dataset.Y)

        return self.tr_x, self.tr_y

    # Caricamento del dataset di test

    def load_test(self):

        test_dataset = CDataset.load(self._pathname + "test_set.gz")

        self.ts_x = CArray.tondarray(test_dataset.X)
        self.ts_y = CArray.tondarray(test_dataset.Y)

        return self.ts_x, self.ts_y

    # Caricamento del dataset completo

    def load_all(self):

        test_dataset = CDataset.load(self._pathname + "test_set.gz")
        train_dataset = CDataset.load(self._pathname + "training_set.gz")

        self.ts_x = CArray.tondarray(test_dataset.X)
        self.ts_y = CArray.tondarray(test_dataset.Y)

        self.tr_x = CArray.tondarray(train_dataset.X)
        self.tr_y = CArray.tondarray(train_dataset.Y)

        print(self.tr_x.shape, self.ts_x.shape)
        print(self.tr_y.shape, self.ts_y.shape)

        tot_dataset = np.vstack((self._tr_x, self._ts_x))
        tot_labels = np.concatenate((self._tr_y, self._ts_y))

        return tot_dataset, tot_labels

    # Caricamento del dataset per k-fold

    def load_for_kfold(self, k_tot, curr_k):

        # Carico il dataset di training

        self.load_training()

        # Calcolo il numero (o lunghezza) di ogni fold

        split_length = int(self.tr_x.shape[0] / k_tot)

        # Creo gli arrey che conterranno il dataset di training e test per ogni fold

        train_dataset = np.empty((0, 2000), int)
        test_dataset = np.empty((0, 2000), int)
        train_labels = np.empty(0, int)
        test_labels = np.empty(0, int)

        # Creo i dataset per ogni fold

        for i in range(k_tot):

            if i == curr_k:

                test_dataset = np.vstack((test_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                test_labels = np.hstack((test_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

            else:

                train_dataset = np.vstack((train_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                train_labels = np.hstack((train_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

        # Riassegno gli attributi di classe

        self.tr_x = train_dataset
        self.tr_y = train_labels
        self.ts_x = test_dataset
        self.ts_y = test_labels

        return self.tr_x, self.tr_y, self.ts_x, self.ts_y

    # Funzione di test per verificare il funzionamento della k-fold 
    # TODO: da rimuovere


'''

    def load_for_kfold_custom(self, tr_ds, tr_l, k_tot, curr_k):

        # Carico il dataset di training
            
        self.tr_x = tr_ds
        self.tr_y = tr_l

        # Calcolo il numero (o lunghezza) di ogni fold

        split_length = int(self.tr_x.shape[0] / k_tot)

        # Creo gli arrey che conterranno il dataset di training e test per ogni fold

        train_dataset = np.empty((0, 2000), int)
        test_dataset = np.empty((0, 2000), int)
        train_labels = np.empty(0, int)
        test_labels = np.empty(0, int)

        # Creo i dataset per ogni fold

        for i in range(k_tot):

            if i == curr_k:

                test_dataset = np.vstack((test_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                test_labels = np.hstack((test_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

            else:

                train_dataset = np.vstack((train_dataset, self.tr_x[i * split_length: (i + 1) * split_length, :]))
                train_labels = np.hstack((train_labels, self.tr_y[i * split_length: (i + 1) * split_length]))

        # Riassegno gli attributi di classe

        self.tr_x = train_dataset
        self.tr_y = train_labels
        self.ts_x = test_dataset
        self.ts_y = test_labels

        return self.tr_x, self.tr_y, self.ts_x, self.ts_y
    
'''
