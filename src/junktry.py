from jcamp import JCAMP_reader
import pandas as pd
import matplotlib.pyplot as plt



def load_data():
    data_jdx = []                                           # empty list initialization for JDX data
    data_csv = []                                           # empty list initialization fro CSV data
    filename = r'E:\inspec_python\test_files\67-63-0-IR.jdx'               # # ask user to select the file to load
    if filename.endswith('.jdx'):                           # check the extension of the file.
        jcamp_dict = JCAMP_reader(filename)
        wavelength = pd.DataFrame(jcamp_dict['x'])
        data = pd.DataFrame(jcamp_dict['y'])
        data_jdx = [wavelength,data]

if __name__=="__main__":
    load_data()