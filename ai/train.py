'''
Disturbance Tracker - Model Trainer

Handles data loading, augmentation, training, validation, and reporting
for training a PyTorch-based neural network to classify audio clips.
'''
import logging
import pathlib
import audiomentations
import sklearn
import torch
import tqdm

# DTrack
import ai.model
import ai.options


class AudioDataset(torch.utils.data.Dataset):
    '''
    Custom PyTorch Dataset for loading and processing audio files.
    '''
    def __init__(self, file_paths, labels, augmentations=None):
        '''
        Initializes the dataset.
        '''
        # List of paths to the audio files
        self.file_paths = file_paths
        # Corresponding list of labels (0 or 1)
        self.labels = labels
        # Audiomentations Compose object
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio using the function from model.py
        audio = ai.model.open_audio_file(filepath)

        # Apply augmentations if specified
        if self.augmentations:
            audio = self.augmentations(
                    samples=audio, sample_rate=ai.model.SAMPLE_RATE)

        # Convert audio to a spectrogram
        spectrogram = ai.model.audio_to_spectrogram(audio)

        return spectrogram, torch.tensor(label, dtype=torch.float32)


def train_all():
    '''
    Main entry point for the training script.

    Train all configured models using tagged audio clips.
    '''
    options = ai.options.bootstrap()
    models_dir = pathlib.Path(options['workspace']) / 'models'

    for model_name in options['inspect_models']:
        logging.info('Begin training: %s', model_name)
        try:
            train_model(model_name, options)
            logging.info('Finished training %s', model_name)
        except KeyboardInterrupt:
            logging.info('Received Ctrl+C; Training stopped')

        ai.model.convert(
                models_dir / f'{model_name}.pth',
                models_dir / f'{model_name}.onnx')
        logging.info('MODEL PREPARED: %s', model_name)


def train_model(model_name, options):
    '''
    Conducts the full training and validation process for a model.
    '''
    workspace = pathlib.Path(options['workspace'])
    models_dir = workspace / 'models'
    tags_dir = workspace / 'tags'
    match_dir = tags_dir / model_name
    empty_dir = tags_dir / 'empty'
    if not match_dir.exists() or not empty_dir.exists():
        raise OSError(f'Missing data directories for {model_name}.')

    # Prepare Dataset
    logging.debug('Loading and preparing dataset')
    match_files = list(match_dir.glob('*.dat'))
    empty_files = list(empty_dir.glob('*.dat'))
    if not match_files or not empty_files:
        raise OSError('Data directories are missing tagged audio data.')

    all_files = match_files + empty_files
    # labels = * len(match_files) + * len(empty_files)
    labels = [1] * len(match_files) + [0] * len(empty_files)

    train_files, val_files, train_labels, val_labels = \
        sklearn.model_selection.train_test_split(
                all_files, labels, test_size=0.2,
                random_state=42, stratify=labels)
    logging.info(
            'Training samples: %d Validation samples: %d',
            len(train_files), len(val_files))

    # 2. Define Augmentations and Create DataLoaders
    train_dataset = AudioDataset(
            train_files,
            train_labels,
            augmentations=audiomentations.Compose([
                audiomentations.AddGaussianNoise(
                    min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                audiomentations.TimeStretch(
                    min_rate=0.8, max_rate=1.25, p=0.5),
                audiomentations.PitchShift(
                    min_semitones=-4, max_semitones=4, p=0.5),
                ]))
    val_dataset = AudioDataset(
            val_files,
            val_labels,
            augmentations=None)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=options['train_batch_size'],
            shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=options['train_batch_size'],
            shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    # TODO: Load model if one exists
    logging.debug('Creating %s model', model_name)
    dev = ai.model.CUDA_CPU
    model = ai.model.NoiseDetector().to(dev)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options['train_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_worse = 0
    history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            }
    for epoch in range(options['train_epochs']):

        # Training Phase
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm.tqdm(
                train_loader, desc=f'Iteration {epoch} [Train]', leave=False):
            inputs, labels = inputs.to(dev), labels.to(dev).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(
                    val_loader,
                    desc=f'Iteration {epoch} [Check]',
                    leave=False):
                inputs, labels = inputs.to(dev), labels.to(dev).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()

        # Log Results
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / len(train_dataset)
        val_accuracy = 100 * val_correct / len(val_dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        logging.debug(
            'Iteration %d: Train Loss: %.4f, Acc: %.2f%% | '
            'Val Loss: %.4f, Acc: %.2f%%',
            epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        scheduler.step(avg_val_loss)

        # Save best iteration; exit if limits reached
        if avg_val_loss < best_val_loss:
            logging.info('Model improved; saving iteration %s', epoch)
            ai.model.save(model, models_dir / f'{model_name}.pth')
            best_val_loss = avg_val_loss
            epochs_worse = 0
        else:
            logging.debug('Training did not improve (%d)', epochs_worse + 1)
            epochs_worse += 1

        if epochs_worse >= options['train_patience']:
            logging.info('Training patience exhausted without improvement')
            return

    logging.info('Training epochs exhausted; check quality of audio samples!')


if __name__ == '__main__':
    train_all()
