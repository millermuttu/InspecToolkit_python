import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use('TkAgg')
from tkinter import *
import random

class Application(Frame):

    def __init__(self, master=None):

        Frame.__init__(self, master)
        matplotlib.rcParams["figure.figsize"] = [2,6]
        self.data_set = [1,2,3,4,5,6]
        self.initUI()

        # to assign widgets
        self.widget = None
        self.toolbar = None

    def initUI(self):
        self.pack(fill=BOTH, expand=1)

        plotbutton = Button(self, text="Plot Data", command=lambda: self.create_plot(self.data_set))
        plotbutton.place(x=300, y=600)

        quitbutton = Button(self, text="Quit", command=self.quit)
        quitbutton.place(x=400, y=600)


    def create_plot(self, dataset):

        # remove old widgets
        if self.widget:
            self.widget.destroy()

        if self.toolbar:
            self.toolbar.destroy()

        # create new elements

        plt = Figure(figsize=(4, 4), dpi=100)

        a = plt.add_subplot(211)
        a.plot(dataset, '-o', label="Main response(ms)")
        a.set_ylabel("milliseconds")
        a.set_title("plot")

        canvas = FigureCanvasTkAgg(plt, self)

        self.toolbar = NavigationToolbar2Tk(canvas, self)
        #toolbar.update()

        self.widget = canvas.get_tk_widget()
        self.widget.pack(fill=BOTH)

        #self.toolbars = canvas._tkcanvas
        #self.toolbars.pack(fill=BOTH)

        # generate a random list of 6 numbers for sake of simplicity, for the next plot
        self.data_set = random.sample(range(30), 6)


def main():

    root = Tk()
    root.wm_title("generic app")
    root.geometry("800x700+100+100")

    app = Application(master=root)
    app.mainloop()

main()