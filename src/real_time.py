import threading
import time
from tkinter import *
from tkinter import ttk, messagebox

import numpy as np
import tensorflow as tf
from pylsl import StreamInlet, resolve_stream

# https://labstreaminglayer.readthedocs.io/

LABELS = ['A', 'B', 'C']
model_path = '../models/EEGNet_labels_3_accuracy_0.4571428596973419'

class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self._start_predict = False
        self.params()
        self.theme()
        self._thread = threading.Thread(target=self.init_lsl)
        self._thread.start()

    @staticmethod
    def predict(data, model):
        probabilities = model.predict(tf.expand_dims(data, 3))
        predicted_indices = tf.argmax(probabilities, 1)
        return tf.gather(LABELS, predicted_indices)

    def init_lsl(self):
        print("looking for an EEG stream...")
        model = tf.keras.models.load_model(model_path)
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print("EEG stream located !", end="\r")
        data = []
        while True:
            sample, timestamp = inlet.pull_sample()
            data.append(sample)
            if self._start_predict:
                self.predict(data, model)
                self._start_predict = False

    def callback(self):
        self._start_predict = True

    def params(self):
        self.title("Serinity - Lab Streaming Layer")
        self.geometry("600x400")

        mainframe = ttk.Frame(self, borderwidth=5, relief="ridge", width=200, height=100)
        mainframe.grid(column=0, row=0)
        ttk.Label(mainframe, text="test").grid(row=1)
        ttk.Button(mainframe, command=self.callback).grid(row=0)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def theme(self):
        self.tk.call('lappend', 'auto_path', "./awthemes-10.4.0")
        self.tk.call('package', 'require', 'awdark')


if __name__ == '__main__':
    Application().mainloop()
