#!/usr/bin/env python3

import os
#import torch
import numpy as np
import argparse
import tensorflow as tf
from EEGNet import EEGNet

#from mlflow import log_metric, log_param, log_artifacts

LABELS = ['A', 'B']
training_data_path = 'data/train'
validation_data_path = 'data/val'

def getData(path):
    data = []
    labels = []
    for label in LABELS:
        label_dir = os.path.join(path, label)
        for session_file in os.listdir(label_dir):
            filepath = os.path.join(label_dir, session_file)
            file = np.load(filepath)
            data.append(tf.constant(file))
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

    print(training_data[0].shape)
    model = EEGNet(nb_classes=2, Chans=32, Samples=len(training_data))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                    metrics=[tf.keras.metrics.Accuracy()])

    model.fit(training_data, training_labels)

    model.save("EEGNet_accuracy")

