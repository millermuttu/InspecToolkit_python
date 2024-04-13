# -*- coding: utf-8 -*-
"""
Created on 1/6/2021

@author: Mallikarjun sajjan (flyingmuttus)
"""
import os
from tkinter import *

import matplotlib
import scipy.io
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.functions import Functions
from src.read_files import load_data, load_dataset, loadData

matplotlib.use('Qt5Agg')
from tkinter import messagebox, IntVar, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import pandas as pd
import time
import matplotlib.pyplot as plt


# main class for GUI appliaction
class InverseGUI(object):

    def __init__(self, master):

        self.master = master  # to initiate the master

        # defining all the varibales
        self.listbox = None
        self.listbox_set = None
        self.Result = None
        self.indexselected = None
        self.toolbarPlot = None
        self.widgetPlot = None
        self.filename = None
        self.listval = 0  # number of data values in data list
        self.listval_set = 0  # number of data values in data set list
        self.spec_wl_data = []  # spectral data + wavelength from data list
        self.spec_wl_data_set = []  # spectral data +wavelength + label from data set list
        self.listboxitems = []  # list of data in listboxitems
        self.listboxitems_set = []  # list of data set in listboxitems_set
        self.spectra = []  # spectral data from datalist
        self.spectra_set = []  # wavelength from data set list
        self.wavelength = []  # wavelength from datalist
        self.wavelength_set = []  # spectral data from data set list
        self.filename = ''
        self.choice = 1

        self.canvas = Canvas(
            self.master,
            bg="#abc4d2",
            height=722,
            width=1009,
            bd=0,
            highlightthickness=0,
            relief="ridge")
        self.canvas.place(x=0, y=0)
        self.other_func = Functions(self)  # to import all the functions from other_func class
        self.createMenu()  # to initiate createMenu function
        self.createToolbar()  # to initiate createToolbar functio
        self.createFigure()  # to initiate createFigure functio
        self.loadDemoData()

    def createMenu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)
        toolsmenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Tools", menu=toolsmenu)
        toolsmenu.add_command(label="Resize", command=self.other_func.resize)
        toolsmenu.add_command(label="Duplicate...", command=self.other_func.duplicate)
        toolsmenu.add_command(label="Transmittance to Absorbance", command=self.other_func.tx2abs)
        toolsmenu.add_command(label="Absorbance to transmittance", command=self.other_func.abs2tx)
        toolsmenu.add_command(label="Transpose", command=self.other_func.transpose)
        toolsmenu.add_separator()
        toolsmenu.add_command(label="More", command=quit)

        transformmenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Transform", menu=transformmenu)
        transformmenu.add_command(label="Resolution change", command=self.other_func.resolution_change)
        transformmenu.add_command(label="Moving average smoothing", command=self.other_func.moving_average)
        transformmenu.add_command(label="Median filter smoothing", command=self.other_func.median_filter)
        transformmenu.add_command(label="SG smoothing", command=self.other_func.SG_filter)
        transformmenu.add_command(label="Gaussian filter", command=self.other_func.gaussian_filter)
        transformmenu.add_command(label="SNV", command=self.other_func.apply_snv)
        transformmenu.add_command(label="MSC", command=self.other_func.apply_msc)
        transformmenu.add_command(label="Normalize", command=self.other_func.normalize)
        transformmenu.add_command(label="SG derivative", command=self.other_func.SG_deriv)
        transformmenu.add_command(label="Interpolate", command=self.other_func.interpolate)
        transformmenu.add_separator()
        transformmenu.add_command(label="More")

        analysismenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Analysis", menu=analysismenu)
        analysismenu.add_radiobutton(label="PCA", command=self.other_func.pca)
        analysismenu.add_radiobutton(label="PLSR", command=None)
        analysismenu.add_radiobutton(label="Logistic regression", command=self.other_func.LR)
        analysismenu.add_radiobutton(label="KNN", command=self.other_func.KNN)
        analysismenu.add_radiobutton(label="Random forest", command=self.other_func.Random_forest)
        analysismenu.add_command(label="Linear discriminative analysis",
                                 command=self.other_func.linear_discreminate_analysis)
        analysismenu.add_command(label="SVM classification", command=self.other_func.svm_classification)
        analysismenu.add_separator()
        analysismenu.add_command(label="More")

        opermenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Arithmetic Operation", menu=opermenu)
        opermenu.add_command(label="Addition", command=self.addition)
        opermenu.add_command(label="Multiplication", command=None)
        opermenu.add_command(label="Subtraction", command=None)
        opermenu.add_command(label="division", command=None)
        opermenu.add_separator()
        opermenu.add_command(label="More")

        helpmenu = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)

    def createFigure(self):
        fig = Figure(figsize=(5, 5), dpi=100)
        self.plotPtr = fig.add_subplot(111)

        # self.plot_fig(range(1, 129), self.sampleData, 'white')
        # plt.plot(range(1,NUMBER_OF_PIXEL+1), self.sampleData, 'w')

        self.canvas.create_text(
            715, 80,
            text="Graph view",
            fill="#4c27de",
            font=("Abel-Regular", int(24.0)))

        self.canvas_figure = FigureCanvasTkAgg(fig, self.master)
        self.canvas_figure.draw()
        self.canvas_figure.get_tk_widget().place(
            x=468.0, y=100,
            width=504.0,
            height=380)

    def plotFigure(self):

        # destroy cuurent toolbar and widget for plot to refresh the plotarea.
        if self.toolbarPlot:
            self.toolbarPlot.destroy()
        if self.widgetPlot:
            self.widgetPlot.destroy()

        # create figure to plot
        fig = Figure(figsize=(5, 5), dpi=100)
        self.plotPtr = fig.add_subplot(111)

        if self.choice == 1:
            self.plotPtr.clear()
            self.plotPtr.plot(self.wavelength, self.spectra)
            self.plotPtr.set_title('RAW ADC Spectra')
            self.plotPtr.set_xlabel('Wavelength')
            self.plotPtr.set_ylabel('Intensity')
        elif self.choice == 2:
            self.plotPtr.clear()
            for y,label in zip(self.spectra_set.T,self.label):
                self.plotPtr.plot(self.wavelength_set, y, label=label)
            self.plotPtr.legend()
            self.plotPtr.set_title('RAW ADC Spectra')
            self.plotPtr.set_xlabel('Wavelength')
            self.plotPtr.set_ylabel('Intensity')

        self.canvas_figure = FigureCanvasTkAgg(fig, self.master)
        self.toolbarPlot = NavigationToolbar2Tk(self.canvas_figure, self.master)
        # self.toolbarPlot.update()
        # placing the toolbar on the Tkinter window
        self.widgetPlot = self.canvas_figure.get_tk_widget()
        self.widgetPlot.place(x=469.0, y=100,width=504.0,height=380)

    def createToolbar(self):

        self.canvas.create_text(
            476.0, 33.0,
            text="Inspec",
            fill="#ec1a1a",
            font=("RibeyeMarrow-Regular", int(48.0)))

        loaddataButton = Button(self.master, text="Load Data", command=self.import_data)
        loaddataButton.place(
            x=280, y=100,
            width=150,
            height=50)

        loaddata_set_Button = Button(self.master, text="Load Dataset", command=self.import_dataset)
        loaddata_set_Button.place(
            x=280, y=170,
            width=150,
            height=50)

        exportResultButton = Button(self.master, text="Export to excel", command=self.export_excel)
        exportResultButton.place(
            x=280, y=240,
            width=150,
            height=50)

        plotButton = Button(self.master, text="Plot", command=self.plotFigure)
        plotButton.place(
            x=280, y=310,
            width=150,
            height=50)

        infoButton = Button(self.master, text="Info", command=self.other_func.info)
        infoButton.place(
            x=280, y=380,
            width=150,
            height=50)

        # fomButton = Button(toolbar, text="Figure of merit", command=None)
        # fomButton.pack(side=LEFT)
        # simuButton = Button(toolbar, text="Simulation",command=None)
        # simuButton.pack(side=LEFT)
        # overlapCheckbox = Checkbutton(toolbar, text="Overlap ?", variable=self.varoverlap)
        # overlapCheckbox.pack(side=RIGHT, padx=1, pady=1)

        self.canvas.create_text(
            115, 60.0,
            text="Data panel",
            fill="#4c27de",
            font=("Abel-Regular", int(18.0)))

        self.canvas.create_text(
            130.0, 400.0,
            text="Data Set panel",
            fill="#4c27de",
            font=("Abel-Regular", int(18.0)))

        frame_listbox = Frame(self.master)
        frame_listbox.place(x=30, y=80)  # Position of where you would place your listbox
        scrollbarV = Scrollbar(frame_listbox, orient=VERTICAL)
        scrollbarH = Scrollbar(frame_listbox, orient=HORIZONTAL)
        self.listbox = Listbox(frame_listbox, width=25, height=12, bg='#f2f8e1')
        self.listbox.config(yscrollcommand=scrollbarV.set,xscrollcommand = scrollbarH.set)
        self.listbox.bind('<<ListboxSelect>>', self.onselect_listbox)
        scrollbarV.config(command=self.listbox.yview)
        scrollbarH.config(command=self.listbox.xview)
        scrollbarV.pack(side=RIGHT, fill='y')
        scrollbarH.pack(side=BOTTOM,fill='x')
        # self.listbox.place(x=37.0, y=119)
        self.listbox.pack(side = TOP)

        frame_listbox_set = Frame(self.master)
        frame_listbox_set.place(x=30, y=415)  # Position of where you would place your listbox_set
        scrollbar_setV = Scrollbar(frame_listbox_set, orient=VERTICAL)
        scrollbar_setH = Scrollbar(frame_listbox_set, orient=HORIZONTAL)
        self.listbox_set = Listbox(frame_listbox_set, width=25, height=12, bg='#f2f8e1')
        self.listbox_set.config(yscrollcommand=scrollbar_setV.set, xscrollcommand=scrollbar_setH.set)
        self.listbox_set.bind('<<ListboxSelect>>', self.onselect_listbox_set)
        scrollbar_setV.config(command=self.listbox_set.yview)
        scrollbar_setH.config(command=self.listbox_set.xview)
        scrollbar_setV.pack(side=RIGHT, fill='y')
        scrollbar_setH.pack(side=BOTTOM, fill='x')
        # self.listbox_set.place(x=35.0, y=413)
        self.listbox_set.pack(side = BOTTOM)

        resultbar = Frame(self.master, width=720, height=150, bg='#f2f8e1')
        self.canvas.create_text(
            340, 500,
            text="Result:",
            fill="#4c27de",
            font=("Abel-Regular", int(24.0)))
        resultbar.place(x=280.0, y=517)
        # resultbar.pack()

    def onselect_listbox(self, evt):
        w = evt.widget
        if w.curselection():
            index = int(w.curselection()[0])  # to get the index of selected data from data list
            self.indexselected = w.get(index)  # get the name of data list selected
            print(index)
            print(self.indexselected)
            self.choice = 1  # make choice as 1 indicating single data
            onselect_data = self.spec_wl_data[index]
            self.spectra = onselect_data[1]  # select spectra from spec_wl_data
            self.wavelength = onselect_data[0]  # select wavelength from spec_wl_data

    def onselect_listbox_set(self, evt):
        w_set = evt.widget
        if w_set.curselection():
            index = int(w_set.curselection()[0])
            self.indexselected = w_set.get(index)
            print(index)
            print(self.indexselected)
            self.choice = 2  # make choice as 2 indicating dataset
            onselect_data = self.spec_wl_data_set[index]
            # print(onselect_data)
            self.spectra_set = onselect_data[1]
            self.wavelength_set = onselect_data[0]
            self.label = onselect_data[2]

    def About(self):
        print("This is a simple gui for inverese spectra")
        messagebox.showinfo('About', 'GUI for inverse spectra.')

    # function to import data from .CSV and .JDX function for single spectral data
    def import_data(self):
        datafromfile = loadData()
        if len(datafromfile)>0:
            Wavelength = datafromfile[0]
            Data = datafromfile[1]
            self.filename = datafromfile[2]
            if len(datafromfile)>0 and Data.shape[1] > 1:
                for k in range(Data.shape[1]):
                    self.spec_wl_data.append([Wavelength, Data[:, k:k + 1]])  # append data to spec_wl_data
                    self.listboxitems.append(f'{self.filename[:-4]}_{(self.listval + k)}')  # append name of spectra to listboxitems
                    self.listbox.insert(END, self.listboxitems[self.listval + k])  # insert name of spectra to listbox
                self.listval = self.listval + Data.shape[1]  # increment listval by number of individual spectras uploaded.
            elif len(datafromfile)>0 and Data.shape[1] ==1:
                data = datafromfile[1]
                self.filename = datafromfile[2]
                self.spec_wl_data.append([Wavelength, data])  # append data to spec_wl_data
                self.listboxitems.append(f'{self.filename[:-4]}_{str(self.listval + 1)}')  # append name of spectra to listbox
                self.listbox.insert(END, self.listboxitems[self.listval])  # insert name of spectra to listbox
                self.listval = self.listval + 1  # increment listval By 1 since jdx contains only one spectras
        else:
            simpledialog.messagebox.showerror("Error", "Data load failed!")

        # data_jdx, data_csv = load_data()
        # if data_csv != []:  # check if data is from CSV
        #     Wavelength = data_csv[0]  # seperate wavelength and spectra
        #     data = data_csv[1]
        #     self.filename = data_csv[2]
        #     for k in range(data.shape[1]):
        #         self.spec_wl_data.append([Wavelength, data.iloc[:, k:k + 1]])  # append data to spec_wl_data
        #         self.listboxitems.append(f'{self.filename}_{(self.listval + k)}')  # append name of spectra to listboxitems
        #         self.listbox.insert(END, self.listboxitems[self.listval + k])  # insert name of spectra to listbox
        #     self.listval = self.listval + data.shape[1]  # increment listval by number of individual spectras uploaded.
        # elif data_jdx != []:  # check if data is from JDX
        #     Wavelength = data_jdx[0]  # seperate wavelength and spectra
        #     data = data_jdx[1]
        #     self.filename = data_jdx[2]
        #     self.spec_wl_data.append([Wavelength.values, data.values])  # append data to spec_wl_data
        #     self.listboxitems.append(f'{self.filename}_{str(self.listval + 1)}')  # append name of spectra to listbox
        #     self.listbox.insert(END, self.listboxitems[self.listval])  # insert name of spectra to listbox
        #     self.listval = self.listval + 1  # increment listval By 1 since jdx contains only one spectras

    def import_dataset(self):
        dataset_csv = load_dataset()
        # wavelength = dataset_csv[0]
        # data = dataset_csv[1]
        self.filename = dataset_csv[3]
        self.spec_wl_data_set.append(dataset_csv)  # append data to spec_wl_data_set
        self.listboxitems_set.append(f'Dataset_{self.filename[:-4]}')  # append name of data set to listboxitems set
        self.listbox_set.insert(END,self.listboxitems_set[self.listval_set])  # insert the name of data set to listboxset
        self.listval_set = self.listval_set + 1  # increment listval_set by 1

    # function to export data from listbox and listboxsset to excel
    def export_excel(self):
        if self.choice == 1:
            wavelength = self.wavelength
            data = self.spectra
            data.insert(column=0, value=wavelength.values.ravel(), loc=0, allow_duplicates=1)
            filename = filedialog.askdirectory()  # ask user to chose the directory to save the excel
            filename = filename + '/' + 'result.xlsx'
            with pd.ExcelWriter(filename) as writer:
                # data.to_excel(writer)
                # wavelength.to_excel(writer)
                # label.to_excel(writer)
                data.to_excel(writer)
            messagebox.showinfo("Success", "spectra stored in result.xlsx file under selected directory")
        else:
            wavelength = self.wavelength_set
            data = self.spectra_set
            label = self.label
            data.insert(value=wavelength.values.ravel(), loc=0, allow_duplicates=1,
                        column=0)  # insert wavelength at coulmn 0 of data
            label.insert(value=[0], loc=0, allow_duplicates=1, column=0)  # insert 0 at (0,0) of label
            mf = [label, data]
            result = pd.concat(mf, join='inner')  # concat mf innerly
            filename = filedialog.askdirectory()  # ask user to chose the directory to save the excel
            filename = filename + '/' + 'result.xlsx'
            with pd.ExcelWriter(filename) as writer:
                # data.to_excel(writer)
                # wavelength.to_excel(writer)
                # label.to_excel(writer)
                result.to_excel(writer)
            messagebox.showinfo("Success",
                                "spectra stored in result.xlsx file under selected directory")  # info to user once saving is complete

    def plot_fig(self, x, data):
        # if self.varoverlap.get() == 0:
        #     plt.clf()
        plt.clf()
        plt.plot(x, data)
        plt.legend()
        plt.title('RAW ADC Spectra')
        plt.xlabel('Wavelength in micrometer')
        plt.ylabel('Raw ADC')

    def plot_fig_dataset(self, x, data, label):
        plt.clf()
        plt.plot(x, data, label=label)
        plt.legend()
        plt.title('RAW ADC Spectra')
        plt.xlabel('Wavelength in micrometer')
        plt.ylabel('Raw ADC')

    def loadDemoData(self):

        filepath = "../demodata/set.mat"
        if os.path.isfile(filepath):
            # loading dataset for demo
            mat = scipy.io.loadmat(filepath)
            wavelength = mat['label']
            label = mat['Y']
            data = mat['X'].T
            self.filename = 'Demo'
            self.spec_wl_data_set.append([wavelength, data, label,self.filename])  # append data to spec_wl_data
            self.listboxitems_set.append(f'Dataset_{str(self.filename)}')  # append name of data set to listboxitems set
            self.listbox_set.insert(END,
                                    self.listboxitems_set[self.listval_set])  # insert the name of data set to listboxset
            self.listval_set = self.listval_set + 1  # increment listval_set by 1

            # Loading individual data for demo
            self.filename = 'Data_demo'
            self.spec_wl_data.append([wavelength, data[1]])  # append data to spec_wl_data
            self.listboxitems.append(f'{self.filename}')  # append name of spectra to listbox
            self.listbox.insert(END, self.listboxitems[self.listval])  # insert name of spectra to listbox
            self.listval = self.listval + 1  # increment listval By 1 since jdx contains only one spectras

    # incomplete function
    def addition(self):
        # if self.choice==1:
        self.spectra = None
        self.wavelength = None
        messagebox.showinfo("Info", "please select first spectra")
        while (self.indexselected == None):
            if self.indexselected != None:
                w1 = self.wavelength
                data1 = self.spectra
                break
        self.spectra = None
        self.wavelength = None
        messagebox.showinfo("info", "please select the second spectra")
        while (self.indexselected != None):
            w2 = self.wavelength
            data2 = self.spectra
        plt.plot(w1, data1)
        plt.plot(w2, data2)


def quit():
    # check if saving
    # if not:
    window.destroy()


if __name__ == "__main__":
    start_time = time.time()
    window = Tk()
    logo = PhotoImage(file='gui/images/iconbitmap.gif')
    window.call('wm', 'iconphoto', window._w, logo)
    window.title("Inspec spectrometer toolkit 1.0.0")
    window.geometry("1009x722")
    window.configure(bg="#abc4d2")

    demo1 = InverseGUI(window)

    window.resizable(False, False)
    window.protocol('WM_DELETE_WINDOW', quit)  # window is your window window

    window.resizable(False, False)
    window.mainloop()
    print("--- %s seconds ---" % (time.time() - start_time))
