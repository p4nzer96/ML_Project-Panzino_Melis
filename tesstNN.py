import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

import numpy as np
from tensorflow.python.keras import callbacks

from tensorflow.python.keras.callbacks import ModelCheckpoint

from LoadDataset import LoadDataset

# Carichiamo il dataset di training e validation

ds_path = "C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\"
load_dataset = LoadDataset(ds_path)

accuracies = np.zeros((4, 100))
losses = np.zeros((4, 100))
val_accuracies = np.zeros((4, 100))
val_losses = np.zeros((4, 100))

for k in range(4):

    train_x, train_y, val_x, val_y = load_dataset.load_for_kfold(4, k)

    # Nel caso ci servissero separo i genuine dai malicious

    genuine_tr_x = train_x[train_y == 1]
    malicious_tr_x = train_x[train_y == 0]

    # Filtraggio del dataset (SOLO PERMISSIONS)

    # Imposto i filtri

    index = 0
    index_list = []

    with open("C:\\Users\\andre\\Desktop\\ML Project - Panzino Melis\\Dataset\\features.txt") as f:

        line_count = 0
        for line in f:
            if line != "\n" and "req_perm" in line:
                line_count += 1
                index_list.append(index)
            index += 1
        f.close()

    # Filtro il dataset

    f_train_x = np.zeros(shape=train_x.shape)
    f_val_x = np.zeros(shape=val_x.shape)

    for item in index_list:
        # Sto prendendo solo i valori presenti a determinati indici

        f_train_x[:, item] = train_x[:, item]
        f_val_x[:, item] = val_x[:, item]

    # %% MODELLO 2

    Training_Set = train_x
    Training_Labels = train_y

    Validation_Set = val_x
    Validation_Labels = val_y

    INPUT_SIZE = len(Training_Set)

    N_EPOCHS = 100
    BATCH_SIZE = 2000

    model = Sequential()

    checkpoint = ModelCheckpoint('CNN_Model_fold{}.h5'.format(k), verbose=1, save_best_only=True)
    # early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.add(Dense(units=INPUT_SIZE, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x=Training_Set,
        y=Training_Labels,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(Validation_Set, Validation_Labels),
        verbose=1,
        callbacks=[checkpoint]
    )

    model.summary()

    accuracies[k, :] = history.history['accuracy']
    losses[k, :] = history.history['loss']
    val_accuracies[k, :] = history.history['val_accuracy']
    val_losses[k, :] = history.history['val_loss']

results = pd.DataFrame(np.vstack(accuracies, losses, val_accuracies, val_losses), columns=[ep for ep in range(1, 101)])
results.to_csv("NN_results.csv")

'''fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(history.history['accuracy'], label="accuracy")
ax[0].set_ylim((0, 1))
ax[0].plot(history.history['val_accuracy'], label = "validation accuracy")
ax[0].set_xlabel("accuracies")
ax[0].legend()
ax[1].plot(history.history['loss'], label="loss")
ax[1].set_xlabel("losses")
ax[1].plot(history.history['val_loss'], label = "validation loss")
ax[1].legend()'''

# %% MODELLO 1

'''
# Impostazione della rete neurale

N_TRAIN_SAMPLES = len(train_y)
N_VAL_SAMPLES = len(val_y)

N_ROWS = 50
N_COLUMNS = 40

Training_Set = np.empty((N_TRAIN_SAMPLES, N_ROWS, N_COLUMNS))  # creo la struttura del dataset
Training_Labels = np.empty(N_TRAIN_SAMPLES)  # creo la struttura dell'array di label

Validation_Set = np.empty((N_VAL_SAMPLES, N_ROWS, N_COLUMNS))  # creo la struttura del dataset
Validation_Labels = np.empty(N_VAL_SAMPLES)  # creo la struttura dell'array di label

nn_train_x = f_train_x
nn_train_y = train_y

nn_val_x = f_val_x
nn_val_y = val_y

for x in range(N_TRAIN_SAMPLES):

    Training_Set[x, :, :] = np.reshape(nn_train_x[x, :], (N_ROWS, N_COLUMNS))
    Training_Labels[x] = nn_train_y[x]

for x in range(N_VAL_SAMPLES):

    Validation_Set[x, :, :] = np.reshape(nn_val_x[x, :], (N_ROWS, N_COLUMNS))
    Validation_Labels[x] = nn_val_y[x]

Training_Set = tf.compat.v2.convert_to_tensor(Training_Set)
Training_Set = tf.expand_dims(Training_Set, -1)

Validation_Set = tf.compat.v2.convert_to_tensor(Validation_Set)
Validation_Set = tf.expand_dims(Validation_Set, -1)

# Creo il modello della rete neurale

N_EPOCHS = 50
BATCH_SIZE = 10000

model = Sequential()

model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(N_ROWS, N_COLUMNS, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

checkpoint = ModelCheckpoint('CNN_{}{}.h5'.format(N_ROWS, N_COLUMNS),
                                verbose=1, save_best_only=True)
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.summary()

history = model.fit(
    x=Training_Set,
    y= Training_Labels,
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(Validation_Set, Validation_Labels),
    verbose=1,
    callbacks=[checkpoint, early]
)

'''
