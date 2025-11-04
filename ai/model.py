'''
Disturbance Tracker - Machine Learning

Trains a machine learning model to detect noise in 2-second audio samples.
Audio format: PCM s16le, 48kHz, mono, 192000 bytes per file
'''
import logging
import pathlib
import numpy as np
import scipy
import joblib


# Frequency bands used for machine learning features
FEATURE_FREQUENCIES = [
        # Basic
        (0, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
        # Low through Mid
        (500, 5000),
        # High
        (2000, 8000),
        ]


def load(path):
    '''
    Returns a trained model
    '''
    if not path.exists():
        logging.debug('No model at %s; loading skipped', path)
        return None
    logging.debug('Loading model from to %s', path)
    return joblib.load(path)


def save(model, path):
    '''
    Saves model to a (pkl) file in workspace directory
    '''
    joblib.dump(model, path)
    logging.info('Model saved to %s', path)


def prepare_dataset(match_directory, empty_directory, augment=False):
    '''
    Returns a prepared dataset with optional data augmentation (fuzzing)
    '''
    x, y = [], []
    for directory, y_append in [(match_directory, 1), (empty_directory, 0)]:
        logging.debug('Loading samples from: %s', directory)
        files = list(pathlib.Path(directory).glob('*.dat'))
        for i, filepath in enumerate(files):
            if filepath.is_file():
                try:
                    audio = normalize_audio(open_file(filepath))
                except OSError as e:
                    logging.error('Error processing %s: %s', filepath, e)
                    continue

                samples = shift_timing(audio) if augment else [audio]
                for audio_sample in samples:
                    features = extract_features(audio_sample)
                    x.append(features)
                    y.append(y_append)

                if (i + 1) % 10 == 0:
                    logging.trace('Processed %s/%s files', i+1, len(files))

    return (np.array(x), np.array(y))


def open_file(filepath):
    '''
    Returns normalized audio from audio file
    '''
    with open(filepath, 'rb') as fh:
        return fh.read()


def normalize_audio(pcm_data):
    '''
    Returns numpy array from (192000 byte) raw PCM audio file
    '''
    audio_data = np.frombuffer(pcm_data, dtype=np.int16)
    return audio_data.astype(np.float32) / 32768.0


def shift_timing(audio_data, num_shifts=5):
    '''
    Returns a list with audio_data and multiple time-shifted (+/-) versions.
    '''
    shifted_versions = [audio_data]
    max_shift = len(audio_data) // 10

    for i in range(1, num_shifts):
        shift_amount = int((i / num_shifts) * max_shift)
        shifted = np.roll(audio_data, shift_amount)
        shifted_versions.append(shifted)

        shifted_neg = np.roll(audio_data, -shift_amount)
        shifted_versions.append(shifted_neg)

    return shifted_versions


def extract_features(audio, sample_rate=48000):
    '''
    Returns extracted features (scipy.signal.spectrogram) from audio data
    '''
    features = []

    features.append(np.mean(audio))
    features.append(np.std(audio))
    features.append(np.max(np.abs(audio)))
    features.append(np.percentile(np.abs(audio), 95))

    # Zero Crossings
    features.append(np.sum(np.diff(np.sign(audio)) != 0) / len(audio))

    fft_vals = np.abs(scipy.fft.fft(audio))
    freqs = scipy.fft.fftfreq(len(audio), 1 / sample_rate)

    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]

    # Spectral_Centroid
    if np.sum(positive_fft) > 0:
        features.append(
                np.sum(positive_freqs * positive_fft) / np.sum(positive_fft))
    else:
        features.append(0)

    # Peak Freqency
    if np.max(positive_fft) > 0:
        features.append(positive_freqs[np.argmax(positive_fft)])
    else:
        features.append(0)

    # All Frequencies
    for low, high in FEATURE_FREQUENCIES:
        band_mask = (positive_freqs >= low) & (positive_freqs <= high)
        band_energy = np.sum(
                positive_fft[band_mask]) / (np.sum(positive_fft) + 1e-10)
        features.append(band_energy)

    _, _, sxx = scipy.signal.spectrogram(
            audio, fs=sample_rate, nperseg=1024)

    # Specral Flux and Rolloff
    if sxx.size > 0:
        features.append(np.mean(np.diff(sxx, axis=1)**2))
        features.append(np.percentile(positive_fft, 85))
    else:
        features.append(0)
        features.append(0)

    energy = np.sum(audio ** 2) / len(audio)
    features.append(energy)

    return np.array(features)
