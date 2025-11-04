'''
Disturbance Tracker - Inspection Utility
'''
import logging
import pathlib
import sys

# DTrack
import ai.options
import ai.model


def main():
    '''
    Entry point for `python3 -m ai.inspect`:
    Perform inference with a specified model against a given input
    '''
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])

    # Load configured models
    models = {}
    for model_name in options['inspect_models']:
        models[model_name] = ai.model.load(workspace / f'{model_name}.pkl')

    # Identify input source (stdin -> file -> directory)
    if not options['inspect_path']:
        logging.trace('Running inference with standard input')
        audio = sys.stdin.buffer.read()
        print(infer_all(models, audio))
    else:
        inspect_path = pathlib.Path(options['inspect_path'])
        if inspect_path.is_file():
            logging.trace('Running inference with single file')
            for audio in slice_audio(inspect_path):
                print(infer_all(models, audio))

        elif inspect_path.is_dir():
            logging.trace('Running inference with directory of mkv files')
            for filename in inspect_path.glob('*.*'):
                logging.info('Reviewing %s', filename)
                for audio in slice_audio(filename):
                    print(infer_all(models, audio))

        else:
            raise OSError(f'Could not find {inspect_path}')


def infer_all(models, pcm_data):
    '''
    Run inference using list of trained models
    '''
    for model_name, model in models.items():
        return (model_name, infer(model, pcm_data))


def infer(model, pcm_data):
    '''
    Run inference using model to determine if audio segment matches
    '''
    normalized_audio = ai.model.normalize_audio(pcm_data)
    features = ai.model.extract_features(normalized_audio)

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]

    return {
        'is_match': bool(prediction),
        'confidence': float(probability[prediction]),
        'probabilities': {
            'miss': float(probability[0]),
            'hit': float(probability[1])}}


def slice_audio(path):
    '''
    Return a list of numpy-prepared 2-second segments from audio file
    '''
    # Test against tagged pcm data files
    if path.match('*.dat'):
        return [ai.model.open_file(path)]

    # TODO: ffmpeg | stdin-to-second | overlapping-to-slices
    if path.match('*.mkv'):
        logging.critical('Not Implemented')
        return []

    logging.warning('Skipping %s (wrong file type)', path)
    return []


if __name__ == '__main__':
    main()
