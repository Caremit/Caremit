"""
Utils to read labeled ECG data (csv), and train a model from that.

Credits to https://github.com/CVxTz/ECG_Heartbeat_Classification
"""

import numpy as np
import pandas as pd
from pathlib import Path

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D
from sklearn.metrics import f1_score, accuracy_score

DATA = Path("~/cloudfiles/code/Data/kaggle-ECG-Heartbeat-Categorization-Dataset/").expanduser()


def load_data(filepath: str):
    """Returns: A four-tuple of numpy arrays:
    signal_data_train, label_data_train, signal_data_full, label_data_full"""
    df_full = pd.read_csv(filepath, header=None)
    print('Full df shape:', df_full.shape)

    # Ensure data is stored as expected: labels are stored in the last column (column 188)
    if not df_full.shape[1] == 188:
        print('Data not formatted as expected, might produce bogus!!')

    # use data subset for training
    df_train = df_full.sample(frac=0.1, random_state=5)  # random seed for reproducibility
    print('Training df shape:', df_train.shape)

    # create np arrays
    signal_data_train = df_train[list(range(187))].to_numpy()
    signal_data_full = df_full[list(range(187))].to_numpy()

    assert np.all(df_full[187] < 255), print('Label codes too large, buggy data??')
    label_data_train = df_train[187].to_numpy().astype(np.int8)
    label_data_full = df_full[187].to_numpy().astype(np.int8)

    return signal_data_train, label_data_train, signal_data_full, label_data_full


def create_cnn(num_features: int = None) -> models.Model:
    """Configures a convolutional neural net with Tensorflow
    Args:
        num_features: number of potential categories to classify the data into:
            how many types of ECG shapes do you expect?
    """
    nclass = num_features or 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(128, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(128, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
                  loss=losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(model: models.Model,
                signal_data: np.array,
                label_data: np.array,
                file_path: str = None,
                epochs: int = None):
    """ file_path: specify where to save the model for future use"""
    early = EarlyStopping(monitor="val_accuracy",
                          mode="max",
                          patience=5,
                          verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_accuracy",
                                  mode="max",
                                  patience=3,
                                  verbose=2)
    callback_list = [early, redonplat]

    if file_path:
        checkpoint = ModelCheckpoint(file_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')
        callback_list.append(checkpoint)

    model.fit(signal_data,
              label_data,
              epochs=epochs or 10,
              verbose=2,
              callbacks=callback_list,
              validation_split=0.1)


def test_model(model: models.Model, test_signal_data: np.array,
               true_label_data: np.array):
    confidence_levels = model.predict(test_signal_data)
    predicted_labels = np.argmax(confidence_levels, axis=-1)

    # verify prediction quality
    f1 = f1_score(y_true=true_label_data, y_pred=predicted_labels, average='macro')
    print(f"F1 score: {f1}")

    accuracy = accuracy_score(y_true=true_label_data, y_pred=predicted_labels)
    print(f"Accuracy score: {accuracy}")
