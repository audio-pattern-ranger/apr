'''
Video List

Navigate list of videos to be reviewed
'''
# Python
import tempfile
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import tkinter.ttk

# APR
import apr.config
import apr.common


class FileSelection(tkinter.ttk.Frame):
    '''
    File Selection "Main Frame"
    '''
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # File selection "globals"
        self.filename = tkinter.StringVar(self)
        self.directory = tkinter.StringVar(self)
        self.directory.set(apr.config.get('workspace') + '/rotating')

        # Directory information
        self.cancel = tkinter.ttk.Button(
                self, text=u'\U0001f870', command=self.cancel,
                width=5, state='disabled')
        self.cancel.grid(row=0, column=0)
        self.refresh = tkinter.ttk.Button(
                self, text=u'\U0001f5d8', command=self.update_list,
                width=5)
        self.refresh.grid(row=0, column=1)
        self.browse = tkinter.ttk.Button(
                self, text=u'\U0001f4c1', command=self.change_directory,
                width=5)
        self.browse.grid(row=0, column=2)
        self.dir_display = tkinter.ttk.Entry(
                self, textvariable=self.directory,
                state='readonly')
        self.dir_display.grid(row=0, column=3, sticky='ew')
        self.open = tkinter.ttk.Button(
                self, text=u'\U0001f872', command=self.open_path,
                width=20, state='disabled')
        self.open.grid(row=0, column=4)

        # Enable back button if current selection exists
        if self.winfo_toplevel().target_file.get():
            self.parent.bind('<Escape>', self.cancel)
            self.cancel.config(state='enabled')

        # File list
        self.packages = tkinter.ttk.Frame(self)
        self.packages.grid(row=1, column=0, columnspan=5, sticky='nsew')
        self.packages.container = None
        self.update_list()

        # Bind hotkeys
        # self.parent.bind('<Return>', self.open_path)

        # Fill entire area
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)

    def update_list(self, event=None):
        # Clear list
        if self.packages.container:
            self.packages.container.destroy()
        self.open.config(state='disabled')

        # Get a list of available files
        self.filelist = apr.common.list_available(self.directory.get())

        # Return early with a basic label/message if none found
        if not self.filelist:
            self.packages.container = tkinter.ttk.Label(
                    self.packages, text='No video or audio files found')
            self.parent.status.config(
                    text='No files (mkv or wav) found in current directory')
            self.filename.set('')
            self.packages.container.pack(expand=True, fill='both')
            return

        # Insert a fresh package list
        self.packages.container = FileList(
                self.packages, sorted(self.filelist))
        self.packages.container.pack(expand=True, fill='both')

        # Update status message
        self.parent.status.config(text='Select a file to review ...')

    def change_directory(self):
        '''
        Provide a directory selection dialog box
        '''
        selected_directory = tkinter.filedialog.askdirectory()
        if selected_directory:
            self.directory.set(selected_directory)
            self.update_list()

    def open_path(self, event=None):
        '''
        Validate selection and trigger reload
        '''
        # Create a temp directory for file review
        self.winfo_toplevel().tempdir = tempfile.TemporaryDirectory()

        # Extract sound into temp directory before processing
        infile = f'{self.directory.get()}/{self.filename.get()}'
        outpath = self.winfo_toplevel().tempdir.name
        apr.common.extract_audio(infile, outpath)

        # Switch to review window
        self.winfo_toplevel().target_file.set(self.filename.get())
        self.winfo_toplevel().target_dir.set(self.directory.get())
        self.winfo_toplevel().set_mainframe('review')

    def cancel(self, event=None):
        '''
        Return to review screen without changing package selection
        '''
        self.winfo_toplevel().set_mainframe('review')


class FileList(tkinter.ttk.Frame):
    '''
    Display a list of pending packages with current status
    '''
    def __init__(self, parent, filelist, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.filelist = filelist

        # File list container
        self.files = tkinter.Listbox(
                self, height=len(self.filelist), selectmode='browse')

        # Add each file to the list
        for file in self.filelist:
            self.files.insert('end', file)

        # Add list to display
        self.files.grid(row=0, column=0, sticky='nsew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Add a scrollbar
        self.scroll = tkinter.ttk.Scrollbar(
                self, style='arrowless.Vertical.TScrollbar')
        self.scroll.configure(command=self.files.yview)
        self.files.configure(yscrollcommand=self.scroll.set)
        self.scroll.grid(row=0, column=1, sticky='ns')

        # Handle selection change
        self.files.bind('<<ListboxSelect>>', self.update_filename)
        self.files.bind('<Double-1>', self.doubleclick)

    def doubleclick(self, event):
        '''
        Trigger parent.open_path via toplevel
        '''
        if not hasattr(self.winfo_toplevel().mainframe.body, 'filename'):
            return None
        self.update_filename(event)
        self.winfo_toplevel().mainframe.body.open_path(event)

    def update_filename(self, event):
        '''
        Update FileSelection.filename from the selected file name
        '''
        if not hasattr(self.winfo_toplevel().mainframe.body, 'filename'):
            return None
        selected_index = self.files.curselection()
        if selected_index:
            pkg_changes = self.files.get(selected_index)  # Get selected file
            self.winfo_toplevel().mainframe.body.filename.set(pkg_changes)
            # Enable package select (open) button
            self.winfo_toplevel().mainframe.body.open.config(state='enabled')
        else:
            # Disable button if we got here without selecting an item
            self.winfo_toplevel().mainframe.body.open.config(state='disabled')
