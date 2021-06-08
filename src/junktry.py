from jcamp import JCAMP_reader
import pandas as pd
import matplotlib.pyplot as plt
import os

import scipy.io

def loadData():
    filepath = r"E:\inspec_python\test_files\AS.csv"
    if filepath.endswith('.jdx'):
        filename = os.path.basename(filepath)
        jcamp_dict = JCAMP_reader(filepath)
        wavelength = pd.DataFrame(jcamp_dict['x']).values
        data = pd.DataFrame(jcamp_dict['y']).values
    elif filepath.endswith('.csv'):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, header=None, sep=',')
        wavelength = df.iloc[:,:1].values
        data = df.iloc[:, 1:].values  # data from all rows and from second column, since first column is wavelength.

    loadedData = [wavelength, data, filename]

    return loadedData

def loadMat():
    filepath = r"E:\Inspec\matlab_version\Inv_spec_gui\DEMO_DATA\AS_Demo.mat"
    mat = scipy.io.loadmat(filepath)
    wavelength = mat['label']
    label = mat['Y']
    data = mat['X'].T
    # self.filename = 'Demo_setdata'
    # self.spec_wl_data.append([wavelength, data,label])  # append data to spec_wl_data
    # self.listboxitems.append(f'{self.filename[:-4]}_{str(self.listval + 1)}')  # append name of spectra to listbox
    # self.listbox.insert(END, self.listboxitems[self.listval])  # insert name of spectra to listbox
    # self.listval = self.listval + 1  # increment listval By 1 since jdx contains only one spectras



if __name__=="__main__":
    loadData()
    filepath = r"E:\Inspec\matlab_version\Inv_spec_gui\DEMO_DATA\set.mat"
    mat = scipy.io.loadmat(filepath)
    wavelength = mat['label']
    label = mat['Y']
    data = mat['X'].T