from tkinter import ttk

from pylsl import StreamInlet, resolve_stream
from EEGNet import EEGNet
import tkinter as tk
import threading

# https://labstreaminglayer.readthedocs.io/

LABELS = ['A', 'B', 'C', 'D', 'E']
training_data_path = 'data/train'
model_path = 'models'
nb_labels = len(LABELS)
nb_electrodes = 32
entries_per_sample = 500  # Nb entries for each sample


def say_hi():
    print("hi there, everyone!")


class Application(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.root = 0

    def callback(self):
        self.root.quit()

    def theme(self):
        self.root.tk.call('lappend', 'auto_path', "C:\\Users\\David\\Desktop\\Serinity\\src\\awthemes-10.4.0")
        self.root.tk.call('package', 'require', 'awdark')
        s = ttk.Style()
        s.theme_names()
        # s.theme_use('aqua')

    def params(self):
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.title("Serinity Lsl")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        feet_entry = ttk.Button(mainframe, width=30)
        feet_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

    def run(self):
        self.root = tk.Tk()
        # self.theme()
        self.params()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        self.root.mainloop()


def lab_streaming_layers(models):
    print("looking for an EEG stream...")
    app = Application()
    return
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    data = []
    while True:
        sample, timestamp = inlet.pull_sample()
        data.append(sample)
        if len(data) > 20000:
            models.predict()
        print(timestamp, sample)


if __name__ == '__main__':
    # model = EEGNet(nb_classes=nb_labels, Chans=nb_electrodes, Samples=entries_per_sample)
    model = 0
    # model.load_weights("../data/model")
    lab_streaming_layers(model)
