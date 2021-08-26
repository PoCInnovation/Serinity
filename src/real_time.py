import threading
import time
from tkinter import *
from tkinter import ttk, messagebox
from tkinter.font import Font

import numpy
import seaborn as sns

import numpy as np
import tensorflow as tf
from pylsl import StreamInlet, resolve_stream

# https://labstreaminglayer.readthedocs.io/

LABELS = ['A', 'B', 'C']
model_path = '../data/models/EEGNet_labels_1_accuracy_1.0'


class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Serinity - Lab Streaming Layer")
        self.geometry("600x400")
        self.iconphoto(False, PhotoImage(file='../public/brain.png'))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._start_predict = False
        self._mainframe = None
        self._state = 0
        self.font = Font(family='Helvetica', size=10)
        self.init_theme()
        self.init_loading()
        try:
            self._thread = threading.Thread(target=self.init_lsl)
            self._thread.start()
        finally:
            return
            if self._state == 0:
                self._progress_bar.stop()
                self._progress_bar["mode"] = "determinate"
                self._progress_bar["value"] = 100
                self._label["text"] = "An error as occured"
            else:
                self.input_button["text"] = "An error as occured"
                self.input_button["state"] = "disabled"

    @staticmethod
    def predict(data, model):

        probabilities = model.predict(tf.expand_dims(data, 2))
        predicted_indices = tf.argmax(probabilities, 1)
        return tf.gather(LABELS, predicted_indices)

    def init_lsl(self):
        print("looking for an EEG stream...")
        model = tf.keras.models.load_model(model_path)
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print("EEG stream located !")
        self.init_main_widget()
        data = []
        while True:
            sample, timestamp = inlet.pull_sample()
            data.append(sample)
            if len(data) > 1000:
                self.input_button["state"] = "normal"
            if self._start_predict:
                prediction = self.predict(data, model)
                self.letter_label["text"] = prediction
                self.sentences_label["text"] += prediction
                if len(self.sentences_label["text"]) % 30 == 0:
                    self.sentences_label["text"] += "\n"
                self._start_predict = False
                self.input_button["state"] = "disabled"
                data = []

    def callback(self):
        self._start_predict = True

    def callback_reset_button(self):
        self.sentences_label["text"] = ""
        self.letter_label["text"] = ""

    def init_main_widget(self):
        self._mainframe.destroy()
        self._state = 1
        self._mainframe = ttk.Frame(self, borderwidth=10, relief="ridge")
        mainframe = self._mainframe
        mainframe.grid(column=0, row=0)
        self.input_button = ttk.Button(mainframe, command=self.callback, text="Predict")
        self.input_button.grid(row=3, pady=20)
        self.input_button["state"] = "disabled"

        self.reset_button = ttk.Button(mainframe, command=self.callback_reset_button, text="Reset")
        self.reset_button.grid(row=4)

        sentences_label_frame = ttk.LabelFrame(mainframe, text="Thought", labelanchor=N)
        sentences_label_frame.grid(row=2, padx=20, pady=25)
        self.sentences_label = ttk.Label(sentences_label_frame, text="HELLO", font=self.font)
        self.sentences_label.grid(column=0, padx=20, pady=10)
        self.letter_label = ttk.Label(sentences_label_frame, text="A", font=Font(family='Helvetica', size=16, weight='bold'))
        self.letter_label.grid(row=1, padx=20, pady=10)

    def init_loading(self):
        self._mainframe = ttk.Frame(self, borderwidth=10, relief="ridge")
        mainframe = self._mainframe
        mainframe.grid(column=0, row=0)
        self._label = ttk.Label(mainframe, text="Looking for an EEG stream", font=self.font)
        self._label.grid(row=1)
        self._progress_bar = ttk.Progressbar(mainframe, mode="indeterminate")
        self._progress_bar.grid(row=2)
        self._progress_bar.start(10)

    def init_theme(self):
        self.tk.call('lappend', 'auto_path', "./awthemes-10.4.0")
        self.tk.call('package', 'require', 'awdark')


if __name__ == '__main__':
    Application().mainloop()
