#!/usr/bin/env python3

from eeglib.helpers import Helper
from eeglib.wrapper import Wrapper
import pandas as pd
import numpy as np
import argparse
import time
import os

def load_csv(filename):
    data = pd.read_csv(filename, skiprows=1,
                       usecols=['EEG.Cz', 'EEG.Fz', 'EEG.Fp1', 'EEG.F7',
                                    'EEG.F3', 'EEG.FC1', 'EEG.C3', 'EEG.FC5',
                                    'EEG.FT9', 'EEG.T7', 'EEG.CP5', 'EEG.CP1',
                                    'EEG.P3', 'EEG.P7', 'EEG.PO9', 'EEG.O1',
                                    'EEG.Pz', 'EEG.Oz', 'EEG.O2', 'EEG.PO10',
                                    'EEG.P8', 'EEG.P4', 'EEG.CP2', 'EEG.CP6',
                                    'EEG.T8', 'EEG.FT10', 'EEG.FC6', 'EEG.C4',
                                    'EEG.FC2', 'EEG.F4', 'EEG.F8', 'EEG.Fp2'])

    helper = Helper(data.to_numpy()) # WindowSize ? 256 ?
    return helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script generating dataset (data, label)')
    parser.add_argument('-f', dest='filename',
                        type=str, help='Csv file to create the dataset', required=True)
    parser.add_argument('-o', dest='output_path',
                        type=str, help='Output file for preprocessed data', required = True)
    args = parser.parse_args()
    helper = load_csv(args.filename);
    for eeg in helper:
        filename = os.path.join(args.output_path, "data_" + str(time.strftime("%d-%m-%Y_%H-%M")))
        np.save(filename, eeg.getChannel())
