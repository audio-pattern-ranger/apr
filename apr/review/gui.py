'''
Root Review Window
'''
# Python
import logging
import tkinter
import tkinter.ttk

# APR
import apr.review.menu
import apr.review.filenav
import apr.review.review


class RootWindow(tkinter.Tk):
    '''
    Main Tk window/container for application and runtime.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Primary application frames
        self.mainframes = {
                'review': apr.review.review.VideoReview,
                'filenav': apr.review.filenav.FileSelection,
                }
        self.winfo_toplevel().target_file = tkinter.StringVar(self)
        self.winfo_toplevel().target_dir = tkinter.StringVar(self)
        self.winfo_toplevel().tempdir = None

        # Window properties
        self.title('APR Review')
        self.minsize(800, 550)

        # Window style
        self.style = tkinter.ttk.Style(self)
        try:
            self.style.theme_use('plastik')
        except tkinter.TclError:
            self.style.theme_use('alt')

        # Main menu
        self.menu = apr.review.menu.MainMenu(self)
        self.config(menu=self.menu)

        # Status bar
        self.status = tkinter.ttk.Label(
                self, relief='sunken', anchor='w')
        self.status.grid(row=1, column=0, sticky='sew')

        # Primary viewport
        self.mainframe = tkinter.ttk.Frame(self)
        self.mainframe.body = None
        self.mainframe.grid(row=0, column=0, sticky='nsew')

        # Initial viewport frame
        self.set_mainframe('filenav')

        # Fill entire area
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def set_mainframe(self, framename):
        '''
        Reset main window with new mainframe.
        '''
        if self.mainframe.body:
            logging.debug('Purging old frame %s', self.mainframe.body)
            self.mainframe.body.grid_forget()
        logging.debug('Loading new frame %s', framename)
        self.mainframe.body = self.mainframes[framename](self)
        self.mainframe.body.grid(row=0, column=0, sticky='nsew')

        # Toggle menu entries
        if framename == 'review':
            self.menu.file.entryconfig('Select File', state='normal')
        elif framename == 'filenav':
            self.menu.file.entryconfig('Select File', state='disabled')
