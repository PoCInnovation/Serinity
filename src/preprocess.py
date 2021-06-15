
from eeglib.helpers import CSVHelper

data_filepath = "./data/Test/"

helper = CSVHelper(data_filepath + "Test_EPOCFLEX_131982_2021.06.08T10.12.3002.00.md.bp.csv")

for eeg in helper:
    print(eeg.PFD())
