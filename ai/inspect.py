'''
Disturbance Tracker - Inspection Utility (Multi-Class)
'''
import json
import logging
import pathlib
import pprint
import subprocess
import sys

# 3rd-Party
import torch
import torch.nn.functional as F

# DTrack project imports
import ai.options
import ai.model


def check_input():
    '''
    Main entry point for the inspection script.
    '''
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])

    # Load models and their labels
    if not options['inspect_models']:
        raise ValueError('No inspection models are configured.')

    loaded_models = {}
    for model_name in options['inspect_models']:
        # Load Labels
        labels_path = workspace / 'models' / f'{model_name}_labels.json'
        if not labels_path.exists():
            raise FileNotFoundError(f'Labels missing for {model_name}')

        with open(labels_path, 'r', encoding='utf-8') as fh:
            labels = json.load(fh)

        # Load Model
        pth_path = workspace / 'models' / f'{model_name}.pth'
        model = ai.model.load(pth_path, num_classes=len(labels))

        loaded_models[model_name] = {
            'model': model,
            'labels': labels}

    # Execution Routing
    if not options.get('inspect_path'):
        # stdin
        logging.debug('Running inference with standard input')
        audio = sys.stdin.buffer.read()
        if not audio:
            raise ValueError('No standard input!')
        pprint.pprint(infer_all(loaded_models, audio))
    else:
        inspect_path = pathlib.Path(options['inspect_path'])
        if inspect_path.is_file():
            # single file
            logging.debug('Running inference with single file')
            check_file(inspect_path, loaded_models)

        elif inspect_path.is_dir():
            # directory of files
            logging.debug('Running inference with directory of mkv files')
            for filename in sorted(inspect_path.glob('*.*')):
                check_file(filename, loaded_models)
        else:
            raise OSError(f'Could not find {inspect_path}')


def check_file(fpath, models):
    '''
    Print results of single-file inspection
    '''
    i = 0
    for audio in slice_audio(fpath):
        i += 1
        for model, prob in infer_all(models, audio).items():
            if prob['match'] == 'empty':
                continue
            logging.debug(prob['distribution'])
            print(f'Matched {model}/{prob["match"]} in {fpath.name} @{i} sec')


def slice_audio(path):
    '''
    Return a list of numpy-prepared 2-second segments from audio file
    '''
    if path.match('*.dat'):
        # Read tagged audio data
        return [ai.model.open_audio_file(path)]
    if path.match('*.mkv'):
        # Read entire audio stream from mkv file
        with subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "warning",
                 "-nostdin", "-nostats", "-i", str(path),
                 "-map", "0:a:0", "-f", "s16le",
                 "-ar", str(ai.model.SAMPLE_RATE), "-ac", "1",
                 "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL) as proc:
            raw_audio = proc.communicate()[0]

        # Slice into 2s segments with 1s overlap using list comprehension
        bps = ai.model.BYTES_PER_SECOND
        return [ai.model.normalize_audio(raw_audio[i:i+ai.model.SAMPLE_SIZE])
                for i in range(0, len(raw_audio)-bps, bps)]

    logging.warning('Skipping %s (wrong file type or not implemented)', path)
    return []


def infer_all(model_bundle, audio_data):
    '''
    Run inference on single audio segment using all trained models.
    Expects model_bundle = {'name': {'model': m, 'labels': [...]}}
    '''
    # Convert raw pcm bytes to numpy array
    if isinstance(audio_data, bytes):
        audio_data = ai.model.normalize_audio(audio_data)

    # Convert the NumPy array into a spectrogram tensor.
    spectrogram = ai.model.audio_to_spectrogram(audio_data)

    # Add a batch dimension (B, C, H, W) for the model
    input_tensor = spectrogram.unsqueeze(0).to(ai.model.CUDA_CPU)

    # Inference Step
    results = {}
    with torch.no_grad():
        for name, data in model_bundle.items():
            model = data['model']
            labels = data['labels']

            # Forward pass (Logits)
            logits = model(input_tensor)

            # Softmax to get probabilities (sum to 1.0)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

            # Handle single class case or list conversion
            if not isinstance(probs, list):
                probs = [probs]

            # Create readable dictionary
            # e.g., {'empty': 0.1, 'barking': 0.9}
            class_probs = {labels[i]: round(probs[i], 4)
                           for i in range(len(labels))}

            # Get best match
            best_idx = torch.argmax(logits, dim=1).item()
            best_label = labels[best_idx]

            results[name] = {
                'match': best_label,
                'confidence': class_probs[best_label],
                'distribution': class_probs}

    return results


if __name__ == '__main__':
    check_input()
