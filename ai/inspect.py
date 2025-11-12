'''
Disturbance Tracker - Inspection Utility
'''
import logging
import pathlib
import pprint
import sys
import json
import torch

# DTrack project imports
import ai.options
import ai.model


def check_input():
    '''
    Main entry point for the inspection script.

    Identify input and return inference results.
    '''
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])

    # Load all models specified in the configuration.
    if not options['inspect_models']:
        raise ValueError('No inspection models are configured.')
    models = {
            model: ai.model.load(workspace / 'models' / f'{model}.pth')
            for model in options['inspect_models']}

    # Execution Routing
    if not options.get('inspect_path'):
        logging.debug('Running inference with standard input')
        audio = sys.stdin.buffer.read()
        if not audio:
            raise ValueError('No standard input!')
        pprint.pprint(infer_all(models, audio))
    else:
        inspect_path = pathlib.Path(options['inspect_path'])
        if inspect_path.is_file():
            logging.debug('Running inference with single file')
            for audio in slice_audio(inspect_path):
                pprint.pprint(infer_all(models, audio))

        elif inspect_path.is_dir():
            logging.debug('Running inference with directory of mkv files')
            for filename in inspect_path.glob('*.*'):
                logging.info('Reviewing %s', filename)
                for audio in slice_audio(filename):
                    pprint.pprint(infer_all(models, audio))

        else:
            raise OSError(f'Could not find {inspect_path}')


def slice_audio(path):
    '''
    Return a list of numpy-prepared 2-second segments from audio file
    '''
    # Test against tagged pcm data files
    if path.match('*.dat'):
        return [ai.model.open_audio_file(path)]

    # TODO: ffmpeg | stdin-to-second | overlapping-to-slices
    if path.match('*.mkv'):
        logging.critical('Not Implemented')
        return []

    logging.warning('Skipping %s (wrong file type)', path)
    return []


def infer_all(models, audio_data):
    '''
    Run inference on single audio segment using all trained models
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
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for model_name, model in models.items():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            is_match = probability > 0.5
            confidence = probability if is_match else 1 - probability

            results[model_name] = {
                'is_match': bool(is_match),
                'confidence': float(confidence),
                }
    return results


if __name__ == '__main__':
    check_input()
