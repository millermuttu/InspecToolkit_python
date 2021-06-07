from tkinter import *


def btn_clicked():
    print("Button Clicked")


window = Tk()
logo = PhotoImage(file='images/iconbitmap.gif')
window.call('wm', 'iconphoto', window._w, logo)
window.title("Inspec spectrometer toolkit 1.0.0")

window.geometry("1009x722")
window.configure(bg = "#abc4d2")
canvas = Canvas(
    window,
    bg = "#abc4d2",
    height = 722,
    width = 1009,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

entry0_img = PhotoImage(file = f"images/img_textBox0.png")
entry0_bg = canvas.create_image(
    115.0, 246.0,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#f2f8e1",
    highlightthickness = 0)

entry0.place(
    x = 37.0, y = 119,
    width = 156.0,
    height = 252)

canvas.create_text(
    476.0, 33.0,
    text = "Inspec",
    fill = "#ec1a1a",
    font = ("RibeyeMarrow-Regular", int(48.0)))

canvas.create_text(
    114.5, 101.0,
    text = "Data",
    fill = "#4c27de",
    font = ("Abel-Regular", int(18.0)))

entry1_img = PhotoImage(file = f"images/img_textBox1.png")
entry1_bg = canvas.create_image(
    113.0, 560.5,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#f2f8e1",
    highlightthickness = 0)

entry1.place(
    x = 35.0, y = 413,
    width = 156.0,
    height = 293)

canvas.create_text(
    115.0, 394.0,
    text = "Data_set",
    fill = "#4c27de",
    font = ("Abel-Regular", int(18.0)))

entry2_img = PhotoImage(file = f"images/img_textBox2.png")
entry2_bg = canvas.create_image(
    720.0, 306.0,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry2.place(
    x = 468.0, y = 115,
    width = 504.0,
    height = 380)

entry3_img = PhotoImage(file = f"images/img_textBox3.png")
entry3_bg = canvas.create_image(
    614.5, 611.5,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#f2f8e1",
    highlightthickness = 0)

entry3.place(
    x = 249.0, y = 517,
    width = 731.0,
    height = 187)

canvas.create_text(
    293.5, 492.0,
    text = "Result:",
    fill = "#4c27de",
    font = ("Abel-Regular", int(24.0)))

canvas.create_text(
    696.5, 93.5,
    text = "Graph view",
    fill = "#4c27de",
    font = ("Abel-Regular", int(24.0)))

img0 = PhotoImage(file =f"images/img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 243, y = 107,
    width = 158,
    height = 50)

img1 = PhotoImage(file =f"images/img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b1.place(
    x = 243, y = 178,
    width = 161,
    height = 50)

img2 = PhotoImage(file =f"images/img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b2.place(
    x = 243, y = 246,
    width = 161,
    height = 50)

img3 = PhotoImage(file =f"images/img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b3.place(
    x = 243, y = 318,
    width = 161,
    height = 50)

window.resizable(False, False)
window.mainloop()
