#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from EEGNet import EEGNet
from sklearn.model_selection import train_test_split

# from mlflow import log_metric, log_param, log_artifacts

LABELS = ['A', 'B', 'C']
training_data_path = 'data/train'
model_path = 'models'

nb_labels = len(LABELS)
nb_electrodes = 32
entries_per_sample = 500  # Nb entries for each sample


def getData(path):
    data = []
    labels = []
    for label_nb, label in enumerate(LABELS):
        label_dir = os.path.join(path, label)
        for session_file in os.listdir(label_dir):
            filepath = os.path.join(label_dir, session_file)
            file = np.load(filepath)
            file = file[:len(file) - len(file) % entries_per_sample]
            print(file.shape)
            splited_data = tf.split(tf.transpose(file), int(len(file) / entries_per_sample), axis=1)
            for sample in splited_data:
                data.append(sample)
                labels.append(tf.constant(label_nb))

    print('Loaded:', len(data), 'samples', '\nShape:', data[0].shape, '\n\n------\n\n')
    return data, labels


if __name__ == "__main__":
    print('Loading training data ...')

    training_data, training_labels = getData(training_data_path)

    training_data, test_data, training_labels, test_labels = train_test_split(training_data, training_labels)

    # log_param("EPOCHS", epochs)
    # log_param("LR", learning_rate)

    model = EEGNet(nb_classes=nb_labels, Chans=nb_electrodes, Samples=entries_per_sample)

    model.compile(loss='categorical_crossentropy', optimizer='Adam',
                  metrics=['accuracy'])

    print(tf.expand_dims(training_data, 3).shape)
    model.fit(tf.expand_dims(training_data, 3),
              tf.keras.utils.to_categorical(training_labels, len(LABELS)), epochs=50)

    loss, accuracy = model.evaluate(tf.expand_dims(test_data, 3),
                                    tf.keras.utils.to_categorical(test_labels, len(LABELS)))

    model.save("models/EEGNet_labels_" + str(len(LABELS)) + "_accuracy_" + str(accuracy))
