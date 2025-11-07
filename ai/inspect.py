'''
Disturbance Tracker - Inspection Utility

This script performs inference on audio files using a trained PyTorch model.
It can process a single file or a directory of files.
'''
import logging
import pathlib
import sys
import torch
import json

# DTrack
import ai.options
import ai.model

def main():
    """
    Entry point for the inspection script.
    """
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load configured models
    models = {}
    for model_name in options['inspect_models']:
        model_path = workspace / f'{model_name}.pth'
        loaded_model = ai.model.load_model(model_path, device)
        if loaded_model:
            models[model_name] = loaded_model

    if not models:
        logging.critical("No models could be loaded. Exiting.")
        return

    # Identify and process input source
    if not options['inspect_path']:
        logging.error("No input path specified. Please use the -i flag.")
        return

    inspect_path = pathlib.Path(options['inspect_path'])
    if inspect_path.is_file():
        logging.trace(f'Running inference on single file: {inspect_path}')
        results = infer(models, inspect_path, device)
        print(json.dumps(results, indent=2))

    elif inspect_path.is_dir():
        logging.trace(f'Running inference on directory: {inspect_path}')
        all_results = {}
        for filename in sorted(inspect_path.glob('*.dat')):
            logging.info('Inspecting %s', filename.name)
            all_results[filename.name] = infer(models, filename, device)
        print(json.dumps(all_results, indent=2))
    else:
        raise OSError(f'Input path not found: {inspect_path}')


def infer(models, file_path, device):
    """
    Runs inference on a single audio file using all loaded models.

    Args:
        models (dict): A dictionary of loaded model instances.
        file_path (pathlib.Path): The path to the .dat audio file.
        device (torch.device): The device to run inference on.

    Returns:
        dict: A dictionary containing the prediction results from each model.
    """
    try:
        # Preprocess the audio file once
        audio_data = ai.model.open_audio_file(file_path)
        spectrogram = ai.model.audio_to_spectrogram(audio_data)
        # Add a batch dimension and send to device
        input_tensor = spectrogram.unsqueeze(0).to(device)

    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
        return {"error": "File processing failed"}

    results = {}
    with torch.no_grad():
        for model_name, model in models.items():
            # Get raw model output (logits)
            output = model(input_tensor)
            # Apply sigmoid to get probability
            probability = torch.sigmoid(output).item()

            is_match = probability > 0.5
            confidence = probability if is_match else 1 - probability

            results[model_name] = {
                'is_match': bool(is_match),
                'confidence': float(confidence),
                'probabilities': {
                    'miss': float(1 - probability),
                    'hit': float(probability)
                }
            }
    return results

if __name__ == '__main__':
    main()