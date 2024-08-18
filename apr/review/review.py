'''
Package Review
'''
# Python
import logging
import pathlib
import tkinter
import tkinter.ttk
import moviepy.editor
import PIL
import PIL.ImageTk

# APR
import apr.config
import apr.common


class VideoReview(tkinter.ttk.Frame):
    '''
    Package Review 'Main Frame'
    '''
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.resize_delay = None

        # Load selected video
        target_dir = pathlib.Path(self.winfo_toplevel().target_dir.get())
        self.video = moviepy.editor.VideoFileClip(
                str(target_dir / self.winfo_toplevel().target_file.get()))

        # Available Clips w/ scrollbar
        self.clips = tkinter.ttk.Frame(self)
        self.clips.grid(row=0, column=0, sticky='ns', rowspan=2)
        self.scroll = tkinter.ttk.Scrollbar(
                self.clips, orient='vertical')
        self.cliplist = tkinter.Listbox(
                self.clips, yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.cliplist.yview)
        self.scroll.pack(side='right', fill='y', expand=True)
        self.cliplist.pack(side='left', fill='both', expand=True)

        # Add one entry for each clip
        for clip in apr.common.list_wav(self.winfo_toplevel().tempdir.name):
            i = int(clip.split('.')[0])
            self.cliplist.insert(tkinter.END, str(i))

        # Bind double-click event to load the selected clip
        self.cliplist.bind('<<ListboxSelect>>', self.clip_changed)

        # Clip Actions
        self.actions = tkinter.Frame(self)
        self.actions.grid(row=0, column=1, sticky='ew')
        # Replay Audio
        self.replay_btn = tkinter.Button(
                self.actions, text='< Replay Audio', command=self.play_frame)
        self.replay_btn.pack(side='left', fill='both', expand=True)
        # Tag for Training
        for model in apr.config.get('models'):
            button = tkinter.Button(
                    self.actions, text=f'Tag as {model}',
                    command=lambda m=model: self.tag_as(m))
            button.pack(side='left', fill='both', expand=True)

        # Set up video viewer
        self.player = tkinter.ttk.Frame(self)
        self.player.grid(row=1, column=1, sticky='nesw')
        self.player.image = tkinter.Label(self.player)
        self.player.image.pack(fill='both', expand=True)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Show initial frame
        # self.after(100, self.load_frame, 1)

        # Resize image when window size changes
        self.parent.bind('<Configure>', self.resize)

    def clip_changed(self, event):
        self.load_frame(int(self.cliplist.get(self.cliplist.curselection())))

    def resize(self, event):
        '''
        Resize loaded frame
        '''
        # Cancel any previously scheduled resize
        if self.resize_delay is not None:
            self.after_cancel(self.resize_delay)

        # Schedule a new call to resize_clip after 500ms
        self.resize_delay = self.after(500, self.resize_clip)

    def load_frame(self, frame_pos):
        '''
        Load image from video at <frame_pos> seconds
        '''
        # Load frame from video
        clip_frame = self.video.get_frame(frame_pos)
        self.source_frame = PIL.Image.fromarray(clip_frame)
        self.resize_clip()

        # Play audio from clip
        self.play_frame()

        # Update status message
        self.parent.status.config(
                text='Loaded frame #{frame}  (@ {stamp})  from  {file}'.format(
                    frame=frame_pos,
                    stamp=apr.common.format_time(frame_pos),
                    file=self.winfo_toplevel().target_file.get()))

    def resize_clip(self):
        if not hasattr(self, 'source_frame'):
            return
        # Get player dimensions and original dimensions
        player_size = (
                self.player.image.winfo_width() - 4,
                self.player.image.winfo_height() - 4)
        scale = min(
                player_size[0] / self.source_frame.width,
                player_size[1] / self.source_frame.height)

        try:
            # Resize image
            resized_frame = self.source_frame.resize(
                    (int(self.source_frame.width * scale),
                     int(self.source_frame.height * scale)),
                    PIL.Image.LANCZOS)

            # Display updated image
            self.image = PIL.ImageTk.PhotoImage(resized_frame)
            self.player.image.config(image=self.image)
            self.player.image.update_idletasks()
        except ValueError:
            pass

    def play_frame(self, event=None):
        '''
        Play the audio file for a selected frame
        '''
        frame_pos = int(self.cliplist.get(self.cliplist.curselection()))
        wav = pathlib.Path(
                self.winfo_toplevel().tempdir.name) / f'{frame_pos:04d}.wav'
        apr.common.play_audio(wav)

    def tag_as(self, model):
        '''
        Copy (tag) an audio file to a directory (model)
        '''
        # Determine source path
        frame_pos = int(self.cliplist.get(self.cliplist.curselection()))
        srcdir = self.winfo_toplevel().tempdir.name
        srcpath = f'{srcdir}/{frame_pos:04d}.wav'

        # Determine destination path
        ws = pathlib.Path(apr.config.get('workspace'))
        of = self.winfo_toplevel().target_file.get()
        dstdir = ws / 'train' / model
        dstpath = f'{dstdir}/{of}:F{frame_pos}.wav'

        # Save file to training data
        apr.common.save_as(srcpath, dstpath)
        notice = f'Saved frame #{frame_pos} of {of} to train {model}'
        self.parent.status.config(text=notice)
        logging.info(notice)
