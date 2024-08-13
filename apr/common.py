'''
Common Utilities
'''
import os
import pathlib
import pydub
import pydub.playback
import subprocess
import shutil


def list_available(directory):
    '''
    List *.mkv and *.ogg files in a given directory.
    '''
    path = pathlib.Path(directory)
    extensions = ['.mkv', '.wav']
    return [p.name for p in path.iterdir() if p.suffix in extensions]


def extract_audio(audio_file, output):
    '''
    Split a wav file into 1-second clips.
    '''
    # Extract .wav from .mkv and "fix" sync rate
    if str(audio_file).endswith('.mkv'):
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', audio_file,
                        '-vn', '-c:a', 'pcm_s16le', '-af', 'aresample=async=1',
                        f'{output}/.audio.wav'])
        audio_file = f'{output}/.audio.wav'

    # Load audio file
    audio = pydub.AudioSegment.from_file(audio_file)

    # Split file into short segments
    for i in range(len(audio) // 1000):
        # Extract 1.1 second of audio
        segment = audio[i * 1000:(i + 1.1) * 1000]
        # Export to wav file
        segment.export(f'{output}/{i+1:04d}.wav', format='wav')

    # Purge conversion data
    if os.path.exists(f'{output}/.audio.wav'):
        os.remove(f'{output}/.audio.wav')


def format_time(total_seconds):
    '''
    Return a formatted time (Hours:Minutes:Seconds) from seconds
    '''
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def list_wav(path):
    '''
    List of .wav files in a given directory
    '''
    p = pathlib.Path(path)
    return sorted([f.name for f in p.glob('*.wav')])


def play_audio(path):
    '''
    Play an audio file
    '''
    audio = pydub.AudioSegment.from_wav(path)
    pydub.playback.play(audio)


def save_as(source, destination):
    '''
    Save a source file into our workspace
    '''
    # Ensure output directory exists
    pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)
