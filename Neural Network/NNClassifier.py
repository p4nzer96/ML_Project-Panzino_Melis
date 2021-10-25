import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


class NNClassifier:

    def __init__(self, nn_model=None):

        self.model = nn_model

    def fit(self, train_x, train_y, val_x=None, val_y=None, batch_size=2000, n_tr_epochs=100,
            model_name="NN_Model"):
        # Trasformo l'input in array

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        # Creo i dataset di train e di validation (se inserito)

        Training_Set = train_x
        Training_Labels = train_y

        Validation_Set = val_x
        Validation_Labels = val_y

        if Validation_Set is not None and Validation_Labels is not None:

            Val_Data = (Validation_Set, Validation_Labels)

        else:

            Val_Data = None

        # Determino i parametri di addestramento

        INPUT_SIZE = len(Training_Set)
        N_EPOCHS = n_tr_epochs
        BATCH_SIZE = batch_size

        # Creo il modello

        checkpoint = ModelCheckpoint(model_name + ".h5", verbose=1, save_best_only=True)

        if self.model is None:
            self.model = Sequential()

            self.model.add(Dense(units=INPUT_SIZE, activation='relu'))
            self.model.add(Dropout(rate=0.2))
            self.model.add(Dense(units=100, activation='relu'))
            self.model.add(Dropout(rate=0.2))
            self.model.add(Dense(units=1, activation='sigmoid'))

            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        # Chiamo il fit del modello

        history = self.model.fit(
            x=Training_Set,
            y=Training_Labels,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=Val_Data,
            verbose=1,
            callbacks=checkpoint
        )

        self.model.summary()

        # Ritorno i valori post addestramento

        accuracies = history.history['accuracy']
        losses = history.history['loss']

        if Val_Data is not None:

            val_accuracies = history.history['val_accuracy']
            val_losses = history.history['val_loss']

            return accuracies, losses, val_accuracies, val_losses

        else:

            return accuracies, losses

    def predict(self, test_x, model_name="NN_Model.h5", batch_size=1):

        test_x = np.asarray(test_x, dtype=int)

        if model_name is None:

            model = self.model

        else:

            try:

                model = load_model(model_name)

            except ImportError:

                print("File Not Found, trying to load the default model...")
                model = self.model

            except TypeError:

                print("Could not load the model")
                return

        if model is None:
            print("Unable to load a model")
            return

        y_pred = model.predict(test_x, batch_size)

        '''
        for i in range(0, test_x.shape[0], batch_size):

            bottom_idx = int(i/batch_size)
            upper_idx = int((i+1)/batch_size)

            y_pred[bottom_idx : upper_idx] = model.predict(test_x [bottom_idx : upper_idx], batch_size)

        '''

        return np.round(y_pred, 0).astype(int)
