from jcamp import JCAMP_reader
import os
from tkinter import filedialog
import pandas as pd
from tkinter import messagebox


def load_data():
    """
    # old implementation of data load
    seperate list for jdx and csv
    :return: two lists for csv and jdx data
    """
    data_jdx = []  # empty list initialization for JDX data
    data_csv = []  # empty list initialization fro CSV data
    filepath = filedialog.askopenfilename()  # # ask user to select the file to load
    if filepath.endswith('.jdx'):  # check the extension of the file.
        filename = os.path.basename(filepath)
        jcamp_dict = JCAMP_reader(filepath)
        wavelength = pd.DataFrame(jcamp_dict['x'])
        data = pd.DataFrame(jcamp_dict['y'])
        data_jdx = [wavelength, data, filename]
    elif filepath.endswith('.csv'):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, header=None, sep=',')
        if df.shape[1] < 2:
            messagebox.showinfo('Warning',
                                'Too few cells selected. You must select at least two columns and five rows.')
        else:
            Wavelength = df[0]
            data = df.iloc[:, 1:]  # data from all rows and from second column, since first column is wavelength.
            data_csv = [Wavelength, data, filename]
    return data_jdx, data_csv


def loadData():
    """
    Function to read individual file with single data, or convert a file with multiple data into single data in data panel.
    Below file formats are supported:
    * jdx
    * csv
    more to come
    :return: List of array
    """
    filepath = filedialog.askopenfilename()
    loadedData = []
    if filepath.endswith('.jdx'):
        filename = os.path.basename(filepath)
        jcamp_dict = JCAMP_reader(filepath)
        wavelength = pd.DataFrame(jcamp_dict['x']).values
        data = pd.DataFrame(jcamp_dict['y']).values

        loadedData = [wavelength, data, filename]

    elif filepath.endswith('.csv'):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, header=None, sep=',')
        if df.shape[1] < 2:
            messagebox.showinfo('Warning',
                                'Too few cells selected. You must select at least two columns and five rows.')
        else:
            wavelength = df.iloc[:, :1].values
            data = df.iloc[:, 1:].values  # data from all rows and from second column, since first column is wavelength.

        loadedData = [wavelength, data, filename]

    return loadedData


def load_dataset():
    """
    Function to read the data from csv as a set of data
    csv should be staored in the specified format
    0           | Labels
    Wavelengths | X_data

    and all the data is read as numpy array
    :return: List od numpy array
    """
    dataset_csv = []
    filepath = filedialog.askopenfilename()
    filename = os.path.basename(filepath)
    df = pd.read_csv(filepath, header=None, sep=',')
    if df.shape[1] < 2:
        messagebox.showinfo('Warning',
                            'Too few cells selected. You must select at least two columns and five rows.')
    else:
        Wavelength = df.iloc[1:, :1].values  # wavelength selecting all rows from 1 to end and all the columns till 0 to 1
        data = df.iloc[1:, 1:].values  # data selecting all the rows from 1 to end and all the columns after 1 to end
        label = df.iloc[0:1, 1:].values  # label selecting first row and all the columns from 1 to end
        dataset_csv = [Wavelength, data, label,filename]
    return dataset_csv
