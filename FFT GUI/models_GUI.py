import matplotlib
matplotlib.use('TkAgg')
from tkinter import *
from notebook import *   # window with tabs
from dftModel_GUI_frame import *


root = Tk( ) 
root.title('fft')
nb = notebook(root, TOP) # make a few diverse frames (panels), each using the NB as 'master': 

# uses the notebook's frame
f1 = Frame(nb( )) 
dft = DftModel_frame(f1)


nb.add_screen(f1, "fft")

nb.display(f1)

root.geometry('+0+0')
root.mainloop( )
