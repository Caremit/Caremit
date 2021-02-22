"""
Utils to read labeled ECG data (csv), and train a model from that.

Inspired by https://github.com/CVxTz/ECG_Heartbeat_Classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D
from sklearn.metrics import f1_score, accuracy_score

DATA = Path("~/cloudfiles/code/Data/kaggle-ECG-Heartbeat-Categorization-Dataset/").expanduser()


def load_data(filepath: str, fraction: float = 1,
              fraction_seed: int = None,
              labels_to_remove: List = None,
              crop_to_equal_distribution=False) -> Tuple[np.array, np.array]:
    """Loads csv data and split it into a signal and a label numpy array,
    respectively.
    Args:
        fraction: fraction of csv data to load, allows to speed up testing.
              Set to 1 (full data) by default
        fraction_seed: random seed to ensure same subset of data across tests
            with same fraction
        labels_to_remove: If you want to exclude some labels, supply them here
        crop_to_equal_distribution: If set, the amount of data per category
            will be limited to the amount of the smallest category"""
    df = pd.read_csv(filepath, header=None)

    # Ensure data is stored as expected: labels are stored in the last column (column 188)
    if not df.shape[1] == 188:
        raise ValueError('Data not formatted as expected, might produce bogus!!')

    # remove unwanted data (if any)
    if labels_to_remove:
        df = df.loc[~df[187].isin(labels_to_remove)].copy()

    # unskew data by limiting to amount of minimum category
    if crop_to_equal_distribution:
        min_count = df[187].value_counts().min()

        dfs = []
        for category in df[187].unique():
            sub_df = df.loc[df[187] == category].copy()
            frac = min_count / len(sub_df)
            sub_df = sub_df.sample(frac=frac)
            dfs.append(sub_df)

        df = pd.concat(dfs)

    # Reduce data for faster training during testing
    kwargs = {'frac': fraction}
    if fraction_seed:
        kwargs['random_state'] = fraction_seed
    df = df.sample(**kwargs)  # shuffle to avoid sorted input!

    signal_data = df[list(range(187))].to_numpy()

    if not np.all(df[187] < 255):
        raise ValueError('Label codes too large, buggy data??')

    label_data = df[187].to_numpy().astype(np.int8)

    return signal_data, label_data


def create_cnn(num_features: int = None) -> models.Model:
    """Configures a convolutional neural net with Tensorflow
    Args:
        num_features: number of potential categories to classify the data into:
            how many types of ECG shapes do you expect?
    """
    nclass = num_features or 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    # img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
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


def eval_prediction(confidence_levels, true_labels, print_all_wrong=False):
    """Inspect model prediction results"""
    df = pd.DataFrame(confidence_levels)
    df['predicted_label'] = df.idxmax(axis=1)
    df['true_label'] = true_labels
    df['prediction_correct'] = df['predicted_label'] == df['true_label']
    df['max_confidence'] = df \
                               .loc[:, list(range(5))] \
        .max(axis=1)

    incorrect_df = df.loc[~df['prediction_correct']].copy()
    print(f'Overall correctness: {(1 - len(incorrect_df) / len(df)) * 100:5.2f} %')

    print('\nCorrectness per category:')

    def fm(x):
        return pd.Series(data=len(x.loc[x['prediction_correct']]) / len(x),
                         index=['corr %'])

    res = df.groupby('predicted_label').apply(fm)
    print(res)

    print('\nHighest confidence for wrong predictions per category:')
    dd = incorrect_df \
        .loc[:, ['predicted_label', 'true_label', 'max_confidence']] \
        .copy()

    print(dd.groupby(['predicted_label', 'true_label']).max())

    # show all confidence distribution for wrong predictions
    if print_all_wrong:
        print('\nConfidence distribution for wrong predictions:')
        sorted_df = incorrect_df \
            .sort_values(by='max_confidence', ascending=False)
        sorted_df.reset_index(inplace=True, drop=True)

        for row in sorted_df.itertuples():
            s = f'{str(row[0]).rjust(4)} '
            s += ', '.join([f'{row[idx]:5.9f}' for idx in range(1, 6)])
            s += f'{row.predicted_label}, {row.true_label}, {row.max_confidence:5.9f}'
            print(s)