'''
APR Inspection
'''
import pathlib
import tempfile
import torch

# APR
import apr.common
import apr.options
import apr.model.train


def entry_point():
    inspect_path = pathlib.Path(apr.options.get('inspect_path'))
    if inspect_path.is_file():
        mframes = scan_single(inspect_path)
        print(sorted(mframes))
    elif inspect_path.is_dir():
        for mkv in inspect_path.glob('*.mkv'):
            mframes = scan_single(mkv)
            print(f'{mkv.name}: {sorted(mframes)}')
    else:
        raise Exception(f'Could not find {inspect_path}')


def scan_single(audio_file):
    '''
    Review each audio segment for a match to the trained model
    '''
    audio_path = pathlib.Path(audio_file)
    if not audio_path.exists():
        raise Exception(f'No video was found: {audio_file}')

    classifier = apr.model.train.AudioClassifier()
    tags = [t for t in apr.config.get('models') if t != 'nomatch']

    # Extract 1-second clips
    tempdir = tempfile.TemporaryDirectory()
    apr.common.extract_audio(audio_path, tempdir.name)

    # Review each clip
    # TODO: This does not support multiple tags
    matched_frames = []
    for wav in pathlib.Path(tempdir.name).glob('*.wav'):
        transformed = classifier.load_audio(wav)[0]
        inputs = transformed.unsqueeze(0)
        with torch.no_grad():
            output = classifier.network(inputs).squeeze(1)
            _, prediction = torch.max(output, len(tags))

        # Check the classification result
        if classifier.index2label[prediction.item()] in tags:
            # Keep track of matched frames
            matched_frames.append(int(wav.name.split('.')[0]))

    return matched_frames
