'''
Disturbance Tracker - Machine Learning Model Definition

This module defines the PyTorch-based NoiseDetector class, which handles
audio preprocessing, spectrogram conversion, and the neural network architecture.
'''
import logging
import numpy as np
import torch
import torch.nn as nn
import librosa
import timm

# --- Constants ---
SAMPLE_RATE = 48000
AUDIO_LENGTH_BYTES = 192000
MODEL_NAME = "efficientnet_b0"
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

class NoiseDetector(nn.Module):
    """
    A PyTorch-based neural network model for detecting specific audio events.

    This class encapsulates a pre-trained EfficientNet model, modified for
    binary classification of audio spectrograms. It also includes methods for
    saving and loading the model's state.
    """
    def __init__(self):
        super().__init__()
        # Load the pre-trained EfficientNet model
        self.base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input spectrograms with shape
                              (batch_size, 3, height, width).

        Returns:
            torch.Tensor: The raw output (logits) from the model.
        """
        return self.base_model(x)

def save_model(model, path):
    """
    Saves the model's state dictionary to a .pth file.

    Args:
        model (NoiseDetector): The PyTorch model instance to save.
        path (pathlib.Path): The file path where the model will be saved.
    """
    logging.info('Saving model to %s', path)
    torch.save(model.state_dict(), path)

def load_model(path, device):
    """
    Loads a trained model from a .pth file.

    Args:
        path (pathlib.Path): The path to the saved .pth model file.
        device (torch.device): The device ('cpu' or 'cuda') to load the model onto.

    Returns:
        NoiseDetector: The loaded and initialized model, ready for inference.
                       Returns None if the file doesn't exist.
    """
    if not path.exists():
        logging.error('No model found at %s; could not load.', path)
        return None

    logging.info('Loading model from %s', path)
    model = NoiseDetector()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def open_audio_file(filepath):
    """
    Loads raw PCM audio data from a client-formatted .dat file.

    Pads or truncates the data to ensure it matches AUDIO_LENGTH_BYTES.

    Args:
        filepath (pathlib.Path): The path to the .dat audio file.

    Returns:
        np.ndarray: A 1D NumPy array of normalized audio samples (float32).
    """
    with open(filepath, 'rb') as fh:
        raw_data = fh.read()

    # Pad with silence (zeros) if the clip is too short
    if len(raw_data) < AUDIO_LENGTH_BYTES:
        raw_data += b'\x00' * (AUDIO_LENGTH_BYTES - len(raw_data))
    # Truncate if the clip is too long
    elif len(raw_data) > AUDIO_LENGTH_BYTES:
        raw_data = raw_data[:AUDIO_LENGTH_BYTES]

    # Convert buffer to numpy array and normalize
    audio_data = np.frombuffer(raw_data, dtype=np.int16)
    return audio_data.astype(np.float32) / 32768.0

def audio_to_spectrogram(audio_data):
    """
    Converts a normalized audio array into a 3-channel Mel spectrogram tensor.

    Args:
        audio_data (np.ndarray): The 1D array of audio samples.

    Returns:
        torch.Tensor: A 3-channel tensor representing the spectrogram image,
                      ready for input into the model.
    """
    # Create the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    # Convert power to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize the spectrogram to a 0-1 range
    img = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

    # Stack the single-channel image to create a 3-channel image
    img_3_channel = np.stack([img, img, img])

    return torch.tensor(img_3_channel, dtype=torch.float32)