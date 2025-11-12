'''
Disturbance Tracker - Machine Learning Model Definition

This module defines the PyTorch-based NoiseDetector class, which handles audio
preprocessing, spectrogram conversion, and the neural network architecture.
'''
import logging
import numpy as np
import torch
from torch import nn
import librosa
import skimage

# DTrack
import ai.options


# NOTE: BytesPerSecond from src/ffmpeg/ffmpeg.go
AUDIO_LENGTH_BYTES = 192000
SAMPLE_RATE = 48000

# Magic calculations for skimage
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Use CUDA device if available, or else CPU
CUDA_CPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NoiseDetector(nn.Module):
    '''
    A PyTorch-based neural network model for detecting specific audio events.
    Uses a simple CNN architecture to ensure ONNX/gonnx compatibility.
    '''
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Output (64, 64, 94) - (1/2 size)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2: Output (128, 32, 47) - (1/4 size)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3: Output (256, 16, 23) - (1/8 size)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: Output (512, 8, 12) - (1/16 size)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 5: Output (512, 4, 6) - (1/32 size)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: (Maximum Depth) Output (1024, 2, 3) - (1/64 size)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1)  # Output features 1024 hongi
        )

    def forward(self, x):
        '''
        Defines the forward pass of the model.
        '''
        x = self.features(x)
        return self.classifier(x)


def save(model, path):
    '''
    Saves the model state to a .pth file.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.trace('Saving model to %s', path)
    torch.save(model.state_dict(), path)


def load(path):
    '''
    Returns a a trained model from a .pth file as NoiseDetector object.
    '''
    logging.trace('Loading model from %s', path)
    device = ai.model.CUDA_CPU
    model = NoiseDetector()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def convert(pth, onnx):
    '''
    Convert pytorch .pth model to ONNX (open model) format.
    NOTE: 128, 188 is super critical; don't ask me why.
    '''
    logging.info('Converting %s to %s', pth, onnx)
    model = load(pth)
    torch.onnx.export(
        model, torch.randn(1, 3, 128, 188), onnx,
        input_names=['input'], output_names=['output'],
        verbose=False)


def open_audio_file(filepath):
    '''
    Loads raw PCM audio data from a client-formatted .dat file.
    '''
    with open(filepath, 'rb') as fh:
        return normalize_audio(fh.read())


def normalize_audio(pcm_data):
    '''
    Convert 2-second buffer to numpy array and normalize.
    '''
    audio_data = np.frombuffer(pcm_data, dtype=np.int16)
    return audio_data.astype(np.float32) / 32768.0


def pad_audio(raw_data):
    '''
    Ensure audio slice is an exact size. (should not be needed)
    '''
    # Pad with silence (zeros) if the clip is too short
    if len(raw_data) < AUDIO_LENGTH_BYTES:
        raw_data += b'\x00' * (AUDIO_LENGTH_BYTES - len(raw_data))
    # Truncate if the clip is too long
    elif len(raw_data) > AUDIO_LENGTH_BYTES:
        raw_data = raw_data[:AUDIO_LENGTH_BYTES]
    return raw_data


def audio_to_spectrogram(audio_data):
    '''
    Converts to Power Spectrogram and RESIZES to match
    the model's expected N_MELS dimension (128 rows).
    '''
    # 1. Power Spectrogram (librosa.stft - Simple Power/Magnitude Squared)
    s = np.abs(librosa.stft(
        y=audio_data,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ))**2

    # 2. Resizing (Row downsampling) to match N_MELS (128)
    # Target shape: (N_MELS, existing columns)
    s_resized = skimage.transform.resize(
            s, (N_MELS, s.shape[1]), anti_aliasing=True, preserve_range=True)

    # 3. Convert Power to Decibels
    s_db = librosa.power_to_db(s_resized, ref=np.max)

    # 4. Normalize the spectrogram to a 0-1 range
    img = (s_db - s_db.min()) / (s_db.max() - s_db.min() + 1e-6)

    # 5. Stack the single-channel image to create a 3-channel image
    img_3_channel = np.stack([img, img, img])

    return torch.tensor(img_3_channel, dtype=torch.float32)
