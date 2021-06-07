from jcamp import JCAMP_reader
from tkinter import *
from tkinter import filedialog
import pandas as pd
from tkinter import messagebox

def load_data():
    data_jdx = []                                           # empty list initialization for JDX data
    data_csv = []                                           # empty list initialization fro CSV data
    filename = filedialog.askopenfilename()                 # # ask user to select the file to load
    if filename.endswith('.jdx'):                           # check the extension of the file.
        jcamp_dict = JCAMP_reader(filename)
        wavelength = pd.DataFrame(jcamp_dict['x'])
        data = pd.DataFrame(jcamp_dict['y'])
        data_jdx = [wavelength,data]
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename, header=None, sep=',')
        if df.shape[1] < 2:
            messagebox.showinfo('Warning',
                                'Too few cells selected. You must select at least two columns and five rows.')
        else:
            Wavelength = df[0]
            data = df.iloc[:, 1:]                               # data from all rows and from second column, since first column is wavelength.
            data_csv = [Wavelength,data]
    return data_jdx,data_csv

def load_dataset():
    dataset_csv = []
    filename = filedialog.askopenfilename()
    df = pd.read_csv(filename, header=None, sep=',')
    if df.shape[1] < 2:
        messagebox.showinfo('Warning',
                            'Too few cells selected. You must select at least two columns and five rows.')
    else:
        Wavelength = df.iloc[1:, :1]                            # wavelength selecting all rows from 1 to end and all the columns till 0 to 1
        data = df.iloc[1:, 1:]                                  # data selecting all the rows from 1 to end and all the columns after 1 to end
        label = df.iloc[0:1, 1:]                                # label selecting first row and all the columns from 1 to end
        dataset_csv = [Wavelength,data,label]
    return dataset_csv

