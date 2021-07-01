#!/usr/bin/env python3

import os
#import torch
import numpy as np
import argparse
import tensorflow as tf
from EEGNet import EEGNet

#from mlflow import log_metric, log_param, log_artifacts

LABELS = ['A', 'B', 'C']
training_data_path = 'data/train'
validation_data_path = 'data/val'
model_path = 'models'

def getData(path):
    data = []
    labels = []
    for label in LABELS:
        label_dir = os.path.join(path, label)
        for session_file in os.listdir(label_dir):
            filepath = os.path.join(label_dir, session_file)
            file = np.load(filepath)
            data.append(tf.transpose(file))
            if label == LABELS[0]:
                labels.append(tf.constant(0))
            elif label == LABELS[1]:
                labels.append(tf.constant(1))
    print('Loaded:', len(data), 'samples', '\nShape:', data[0].shape, '\n\n------\n\n')
    return data, labels

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Script training our model from data dir')

    print('Loading training data ...')

    training_data, training_labels = getData(training_data_path)
    #print('Loading validation data ...')
    #validation_data, validation_labels = getData(validation_data_path)

    #SHUFFLE ?

    #log_param("EPOCHS", epochs)
    #log_param("LR", learning_rate)
    nb_labels = len(LABELS)
    nb_electrodes = 32
    samples_every_5_sec = 1312 # Entries every 5 sec

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="models")

    model = EEGNet(nb_classes=nb_labels, Chans=nb_electrodes, Samples=samples_every_5_sec)

    #Binary crossentropy (two labels)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                    metrics=[tf.keras.metrics.Accuracy()])

    model.fit(tf.expand_dims(training_data, 3), tf.keras.utils.to_categorical(training_labels, 2), callbacks=[cp_callback])

    # loss, accuracy = model.evaluate(test_data, test_labels) get accuracy for test dataset

    model.save("models/EEGNet") # model_name + accuracy
