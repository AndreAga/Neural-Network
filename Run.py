import Execution as Exe

from Tkinter import *
import tkFileDialog
import tkMessageBox


class GUI (Tk):

    def __init__(self):

        # ------------- CONFIGURATION GUI  ------------ #

        Tk.__init__(self)

        self.title('CONFIGURATION')

        self.resizable(False, False)
        self.withdraw()
        self.update_idletasks()

        x = (self.winfo_screenwidth() - 1100) / 2
        y = (self.winfo_screenheight() - 500) / 2

        self.geometry("+%d+%d" % (x, y))
        self.deiconify()
        self.grid()

        lbl_dataset_path = Label(self, text="DATASET")
        lbl_dataset_path.grid(column=0, row=0, padx=5, pady=5, columnspan=9)

        self.var_dataset_path = StringVar()
        lbl_dataset_path = Label(self, text="Dataset CSV:")
        lbl_dataset_path.grid(column=0, row=1, padx=10, pady=0, sticky='E')
        txt_dataset_path = Entry(self, textvariable=self.var_dataset_path, width=80)
        txt_dataset_path.grid(column=1, row=1, padx=0, pady=10, columnspan=5)
        bt_dataset = Button(self, text="Select", command=self.select_dataset)
        bt_dataset.grid(column=6, row=1, padx=0, pady=10, sticky='W')

        self.var_dataset_delimiter = StringVar()
        lbl_dataset_delimiter = Label(self, text="CSV Delimiter:")
        lbl_dataset_delimiter.grid(column=7, row=1, padx=10, pady=0, sticky='E')
        txt_dataset_delimiter = Entry(self, textvariable=self.var_dataset_delimiter, width=5)
        txt_dataset_delimiter.grid(column=8, row=1, padx=10, pady=5, sticky='W')
        self.var_dataset_delimiter.set(",")

        self.var_dataset_attributes = IntVar()
        lbl_dataset_attributes = Label(self, text="CSV Columns #:")
        lbl_dataset_attributes.grid(column=0, row=2, padx=10, pady=0, sticky='E')
        txt_dataset_attributes = Spinbox(self, textvariable=self.var_dataset_attributes, from_=1, to=1000, width=7)
        txt_dataset_attributes.grid(column=1, row=2, padx=0, pady=0, sticky='W')

        self.var_dataset_class = IntVar()
        lbl_dataset_class = Label(self, text="Class Column:")
        lbl_dataset_class.grid(column=2, row=2, padx=10, pady=0, sticky='E')
        txt_dataset_class = Spinbox(self, textvariable=self.var_dataset_class, from_=1, to=1000, width=7)
        txt_dataset_class.grid(column=3, row=2, padx=5, pady=5, sticky='W')

        self.var_dataset_shake = IntVar()
        lbl_dataset_shake = Label(self, text="Shake #:")
        lbl_dataset_shake.grid(column=4, row=2, padx=10, pady=0, sticky='E')
        txt_dataset_shake = Spinbox(self, textvariable=self.var_dataset_shake, from_=0, to=10, width=5)
        txt_dataset_shake.grid(column=5, row=2, padx=5, pady=5, sticky='W')
        self.var_dataset_shake.set(5)

        lbl_fs = Label(self, text="FEATURES SELECTION")
        lbl_fs.grid(column=0, row=3, padx=5, pady=5, columnspan=9)

        self.var_fs_type = StringVar()
        self.txt_fs_vt = Radiobutton(self, text="Filter Variance", var=self.var_fs_type, value="VT",
                                     command=self.disable_fs_info)
        self.txt_fs_vt.grid(column=1, row=4, padx=0, pady=0, sticky=W)
        self.txt_fs_f = Radiobutton(self, text="Filter FCBF", var=self.var_fs_type, value="FCBF",
                                    command=self.disable_fs_info)
        self.txt_fs_f.grid(column=1, row=5, padx=0, pady=0, sticky=W)
        self.txt_fs_fs = Radiobutton(self, text="Forward Selection", var=self.var_fs_type, value="FS",
                                     command=self.disable_fs_info)
        self.txt_fs_fs.grid(column=1, row=6, padx=0, pady=1, sticky=W)
        self.txt_fs_skip = Radiobutton(self, text="Skip", var=self.var_fs_type, value="Skip",
                                       command=self.disable_fs_info)
        self.txt_fs_skip.grid(column=1, row=7, padx=0, pady=0, sticky=W)
        self.txt_fs_skip.select()

        self.var_fs_type_vt_min = DoubleVar()
        lbl_fs_type_vt_min = Label(self, text="Variance Min:")
        lbl_fs_type_vt_min.grid(column=2, row=4, padx=0, pady=0, sticky='E')
        self.txt_fs_type_vt_min = Entry(self, textvariable=self.var_fs_type_vt_min, width=7)
        self.txt_fs_type_vt_min.grid(column=3, row=4, padx=0, pady=0, sticky='W')

        self.var_fs_type_vt_max = DoubleVar()
        lbl_fs_type_vt_max = Label(self, text="Variance Max:")
        lbl_fs_type_vt_max.grid(column=4, row=4, padx=0, pady=0, sticky='E')
        self.txt_fs_type_vt_max = Entry(self, textvariable=self.var_fs_type_vt_max, width=7)
        self.txt_fs_type_vt_max.grid(column=5, row=4, padx=0, pady=0, sticky='W')
        self.var_fs_type_vt_max.set(100.0)

        self.var_fs_type_f = DoubleVar()
        lbl_fs_type_f = Label(self, text="Delta SU:")
        lbl_fs_type_f.grid(column=2, row=5, padx=0, pady=0, sticky='E')
        self.txt_fs_type_f = Entry(self, textvariable=self.var_fs_type_f, width=7)
        self.txt_fs_type_f.grid(column=3, row=5, padx=0, pady=0, sticky='W')

        self.var_fs_type_fs = IntVar()
        lbl_fs_type_fs = Label(self, text="Order Features:")
        lbl_fs_type_fs.grid(column=2, row=6, padx=10, pady=0, sticky='E')
        self.txt_fs_type_fs_i = Radiobutton(self, text="Original", var=self.var_fs_type_fs, value=0,
                                            command=self.disable_fs_info)
        self.txt_fs_type_fs_i.grid(column=3, row=6, padx=0, pady=1, sticky=W)
        self.txt_fs_type_fs_i.select()
        self.txt_fs_type_fs_lh = Radiobutton(self, text="Low to High", var=self.var_fs_type_fs, value=1,
                                             command=self.disable_fs_info)
        self.txt_fs_type_fs_lh.grid(column=4, row=6, padx=0, pady=0, sticky=W)
        self.txt_fs_type_fs_hl = Radiobutton(self, text="High to Low", var=self.var_fs_type_fs, value=2,
                                             command=self.disable_fs_info)
        self.txt_fs_type_fs_hl.grid(column=5, row=6, padx=0, pady=1, sticky=W)

        self.var_fs_type_fs_criterion = IntVar()
        lbl_fs_type_fs_criterion = Label(self, text="Condition:")
        lbl_fs_type_fs_criterion.grid(column=6, row=6, padx=10, pady=0, sticky='WE', columnspan=2)
        self.txt_fs_type_fs_criterion_eq = Radiobutton(self, text=">=", var=self.var_fs_type_fs_criterion, value=0,
                                                       command=self.disable_fs_info)
        self.txt_fs_type_fs_criterion_eq.grid(column=7, row=6, padx=0, pady=0, sticky=E)
        self.txt_fs_type_fs_criterion_neq = Radiobutton(self, text=">", var=self.var_fs_type_fs_criterion, value=1,
                                                        command=self.disable_fs_info)
        self.txt_fs_type_fs_criterion_neq.grid(column=8, row=6, padx=0, pady=1, sticky=W)
        self.txt_fs_type_fs_criterion_neq.select()

        self.disable_fs_info()

        lbl_nn = Label(self, text="NEURAL NETWORK")
        lbl_nn.grid(column=0, row=8, padx=5, pady=5, columnspan=9)

        self.var_nn_hidden = IntVar()
        lbl_nn_hidden = Label(self, text="Add Hidden Nodes:")
        lbl_nn_hidden.grid(column=0, row=9, padx=10, pady=0, sticky='E')
        txt_nn_hidden = Spinbox(self, textvariable=self.var_nn_hidden, from_=-100, to=100, width=7)
        txt_nn_hidden.grid(column=1, row=9, padx=0, pady=0, sticky='W')
        self.var_nn_hidden.set(1)

        self.var_nn_epochs = IntVar()
        lbl_nn_epochs = Label(self, text="Epochs:")
        lbl_nn_epochs.grid(column=2, row=9, padx=10, pady=0, sticky='E')
        txt_nn_epochs = Spinbox(self, textvariable=self.var_nn_epochs, from_=0, to=100000, width=7)
        txt_nn_epochs.grid(column=3, row=9, padx=5, pady=5, sticky='W')
        self.var_nn_epochs.set(1000)

        self.var_nn_n = DoubleVar()
        lbl_nn_n = Label(self, text="Learning Rate:")
        lbl_nn_n.grid(column=4, row=9, padx=10, pady=0, sticky='E')
        txt_nn_n = Entry(self, textvariable=self.var_nn_n, width=7)
        txt_nn_n.grid(column=5, row=9, padx=5, pady=5, sticky='W')
        self.var_nn_n.set(0.01)

        self.var_nn_m = DoubleVar()
        lbl_nn_m = Label(self, text="Momentum:")
        lbl_nn_m.grid(column=6, row=9, padx=10, pady=0, sticky='E')
        txt_nn_m = Entry(self, textvariable=self.var_nn_m, width=7)
        txt_nn_m.grid(column=7, row=9, padx=5, pady=5, sticky='W')

        self.var_nn_folds = IntVar()
        lbl_nn_folds = Label(self, text="CrossValidation Folds:")
        lbl_nn_folds.grid(column=0, row=10, padx=10, pady=0, sticky='E')
        txt_nn_folds = Spinbox(self, textvariable=self.var_nn_folds, from_=0, to=1000, width=7)
        txt_nn_folds.grid(column=1, row=10, padx=0, pady=0, sticky='W')
        self.var_nn_folds.set(5)

        self.var_nn_val = IntVar()
        lbl_nn_val = Label(self, text="Validation Set %:")
        lbl_nn_val.grid(column=2, row=10, padx=10, pady=0, sticky='E')
        self.txt_nn_val = Spinbox(self, textvariable=self.var_nn_val, from_=0, to=100, width=7)
        self.txt_nn_val.grid(column=3, row=10, padx=5, pady=5, sticky='W')

        self.var_nn_es = DoubleVar()
        lbl_nn_es = Label(self, text="Early Stop:")
        lbl_nn_es.grid(column=4, row=10, padx=10, pady=0, sticky='E')
        self.txt_nn_es = Entry(self, textvariable=self.var_nn_es, width=7)
        self.txt_nn_es.grid(column=5, row=10, padx=5, pady=5, sticky='W')
        self.var_nn_es.set(0.001)

        self.var_nn_val.trace('w', self.disable_early_stop)
        self.var_nn_val.set(0)

        lbl_exe = Label(self, text="RESULT")
        lbl_exe.grid(column=0, row=11, padx=5, pady=5, columnspan=9)

        self.var_output_path = StringVar()
        lbl_output_path = Label(self, text="Save Result in:")
        lbl_output_path.grid(column=0, row=12, padx=10, pady=0, sticky='E')
        self.txt_output_path = Entry(self, textvariable=self.var_output_path, width=80)
        self.txt_output_path.grid(column=1, row=12, padx=0, pady=10, columnspan=5)
        bt_output_path = Button(self, text="Select", command=self.select_output_path)
        bt_output_path.grid(column=6, row=12, padx=0, pady=10, sticky='W')
        self.txt_output_path.configure(state=DISABLED)

        self.var_output_save = BooleanVar()
        lbl_output_save = Label(self, text="Save Result:")
        lbl_output_save.grid(column=7, row=12, padx=10, pady=0, sticky='E')
        txt_output_save = Checkbutton(self, variable=self.var_output_save, command=self.disable_output_path)
        txt_output_save.grid(column=8, row=12, padx=10, pady=5, sticky='W')
        txt_output_save.select()
        self.disable_output_path()

        self.btn_execute = Button(self, text='Start', command=self.execute)
        self.btn_execute.grid(column=0, row=15, padx=15, pady=10, columnspan=9)

        # -------------------------------------------- #

    # Get Dataset URI and fill output directory
    def select_dataset(self):
        path = tkFileDialog.askopenfilename(parent=self, title='Choose a Dataset')
        if path.strip() != '':
            if self.var_output_path.get().strip() == '':
                slash = path.rfind('/')
                output_path = path[:slash+1]+'Results'
                self.var_output_path.set(output_path)
            self.var_dataset_path.set(path)

    # Get Path for output files
    def select_output_path(self):
        path = tkFileDialog.askdirectory(parent=self, title='Choose an Output Directory')
        if path.strip() != '':
            self.var_output_path.set(path)

    # Disable/Enable output path
    def disable_output_path(self):
        if self.var_output_save.get():
            self.txt_output_path.configure(state=NORMAL)
        else:
            self.txt_output_path.configure(state=DISABLED)

    # Disable/Enable Window components according the features selection
    def disable_fs_info(self):
        if self.var_fs_type.get() == 'VT':
            self.txt_fs_type_vt_min.configure(state=NORMAL)
            self.txt_fs_type_vt_max.configure(state=NORMAL)
            self.txt_fs_type_f.configure(state=DISABLED)
            self.txt_fs_type_fs_i.configure(state=DISABLED)
            self.txt_fs_type_fs_lh.configure(state=DISABLED)
            self.txt_fs_type_fs_hl.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_eq.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_neq.configure(state=DISABLED)
        elif self.var_fs_type.get() == 'FCBF':
            self.txt_fs_type_vt_min.configure(state=DISABLED)
            self.txt_fs_type_vt_max.configure(state=DISABLED)
            self.txt_fs_type_f.configure(state=NORMAL)
            self.txt_fs_type_fs_i.configure(state=DISABLED)
            self.txt_fs_type_fs_lh.configure(state=DISABLED)
            self.txt_fs_type_fs_hl.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_eq.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_neq.configure(state=DISABLED)
        elif self.var_fs_type.get() == 'FS':
            self.txt_fs_type_vt_min.configure(state=DISABLED)
            self.txt_fs_type_vt_max.configure(state=DISABLED)
            self.txt_fs_type_f.configure(state=DISABLED)
            self.txt_fs_type_fs_i.configure(state=NORMAL)
            self.txt_fs_type_fs_lh.configure(state=NORMAL)
            self.txt_fs_type_fs_hl.configure(state=NORMAL)
            self.txt_fs_type_fs_criterion_eq.configure(state=NORMAL)
            self.txt_fs_type_fs_criterion_neq.configure(state=NORMAL)
        else:
            self.txt_fs_type_vt_min.configure(state=DISABLED)
            self.txt_fs_type_vt_max.configure(state=DISABLED)
            self.txt_fs_type_f.configure(state=DISABLED)
            self.txt_fs_type_fs_i.configure(state=DISABLED)
            self.txt_fs_type_fs_lh.configure(state=DISABLED)
            self.txt_fs_type_fs_hl.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_eq.configure(state=DISABLED)
            self.txt_fs_type_fs_criterion_neq.configure(state=DISABLED)

    def disable_early_stop(self, a, b, c):
        if self.var_nn_val.get() == 0:
            self.txt_nn_es.configure(state=DISABLED)
        else:
            self.txt_nn_es.configure(state=NORMAL)

    def execute(self):

        # Check if some fields are filled
        if self.var_dataset_path.get().strip() == '':
            tkMessageBox.showwarning("Dataset", "Insert Dataset Path")
        elif (self.var_output_save.get()) and (self.var_output_path.get().strip() == ''):
            tkMessageBox.showwarning("Output", "Select Directory to store the Result")
        else:

            if self.var_dataset_delimiter.get() == '':
                self.var_dataset_delimiter.set(' ')

            # Get dataset path and dataset name
            slash = self.var_dataset_path.get().rfind('/')
            tmp = self.var_dataset_path.get()[slash+1:]
            point = tmp.rfind('.')

            dataset_name = tmp[:point]
            dataset_path = self.var_dataset_path.get()[:slash]

            # Initialization & Execution
            Exe.Initialization(self.var_output_save.get(), self.var_output_path.get() + '/' + dataset_name,
                               dataset_path, dataset_name, self.var_dataset_delimiter.get(),
                               self.var_dataset_attributes.get(), self.var_dataset_class.get(),
                               self.var_fs_type_vt_min.get(), self.var_fs_type_vt_max.get(),
                               self.var_nn_folds.get(), self.var_dataset_shake.get(),
                               self.var_nn_hidden.get(), self.var_nn_epochs.get(), self.var_nn_es.get(),
                               self.var_nn_n.get(), self.var_nn_m.get(), self.var_nn_val.get(),
                               self.var_fs_type.get(), self.var_fs_type_f.get(), self.var_fs_type_fs.get(),
                               self.var_fs_type_fs_criterion.get())


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
