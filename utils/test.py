from tkinter import *

root = Tk()
root.geometry('500x300')

frame = Frame(root)
frame.place(x = 25, y = 25) # Position of where you would place your listbox

lb = Listbox(frame, width=30, height=6)
lb.pack(side = 'left',fill = 'y' )

scrollbar = Scrollbar(frame, orient="vertical",command=lb.yview)
scrollbar.pack(side="right", fill="y")

lb.config(yscrollcommand=scrollbar.set)

# lb1 = Listbox(frame, width=30, height=6)
# lb1.place(x=40,y=45)
#
# scrollbar = Scrollbar(frame, orient="vertical",command=lb1.yview)
# scrollbar.pack(side="right", fill="y")
#
# lb1.config(yscrollcommand=scrollbar.set)

for i in range(10):
    lb.insert(END, 'test'+str(i))
    # lb1.insert(END, 'test' + str(i))

root.mainloop()