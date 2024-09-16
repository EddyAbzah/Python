# GUI frame for the dftModel_function.py

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import tkFileDialog, tkMessageBox
except ImportError:
    # for Python3
    from tkinter import *  ## notice lowercase 't' in tkinter here
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
import sys, os
from scipy.io.wavfile import read
import dftModel_function
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))


class DftModel_frame:
  
    def __init__(self, parent):  
         
        self.parent = parent        
        self.initUI()

    def initUI(self):

        choose_label = "Input file (.txt, Spi records):"
        Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
        #TEXTBOX TO PRINT PATH OF THE spi FILE
        self.filelocation = Entry(self.parent)
        self.filelocation.focus_set()
        self.filelocation["width"] = 25
        self.filelocation.grid(row=1,column=0, sticky=W, padx=10)
        self.filelocation.delete(0, END)
         # self.filelocation.insert(0, '../../recods/Rec0..._SPI.txt')

        #BUTTON TO BROWSE rec FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
        self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 

        # res_folder = "Result folder:"
        # Label(self.parent, text=res_folder).grid(row=6, column=0, sticky=W, padx=5, pady=(10,2))
        # #TEXTBOX TO PRINT PATH OF THE spi FILE
        # self.result_file = Entry(self.parent)
        # self.result_file.focus_set()
        # self.result_file["width"] = 25
        # self.result_file.grid(row=7,column=0, sticky=W, padx=10)
        # self.result_file.delete(0, END)
        #  # self.result_file.insert(0, '../../results/')
        # #BUTTON TO BROWSE results
        # self.open_folder = Button(self.parent, text="Browse...", command=self.browse_result_folder) #see: def browse_file(self)
        # self.open_folder.grid(row=7, column=0, sticky=W, padx=(220, 6))
        #


        
        Fs = "Fs:"
        Label(self.parent, text=Fs).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
        self.Fs = Entry(self.parent, justify=CENTER)
        self.Fs["width"] = 7
        self.Fs.grid(row=2,column=0, sticky=W, padx=(100,5), pady=(10,2))
        self.Fs.delete(0, END)
        self.Fs.insert(0, "50000")
        


        #BUTTON TO COMPUTE EVERYTHING
        self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
        self.compute.grid(row=9, column=0, padx=5, pady=(10,15), sticky=W)

        # define options for opening file
        self.file_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('All files', '.*'), ('spi files', '.txt')]
        options['initialdir'] = '../../Recs/'
        options['title'] = 'Open a spi recored  file .txt'
        
   

    
    def browse_file(self):
        
        self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
        #set the text of the self.filelocation
        self.filelocation.delete(0, END)
        self.filelocation.insert(0,self.filename)
    



    def compute_model(self):
        
        try:
            inputFile = self.filelocation.get()
            Fs=int(self.Fs.get())

            dftModel_function.main(inputFile,Fs)

        except ValueError as errorMessage:
            tkMessageBox.showerror("Input values error",errorMessage)
            
