'''
Disturbance Tracker - Model Trainer

Uses directory structure: tags/<model_name>/<class>/<audio_segment>.dat
'''
import logging
import pathlib
import json
import audiomentations
import sklearn.model_selection
import torch
import tqdm
import sys

# DTrack
import ai.model
import ai.options


class AudioDataset(torch.utils.data.Dataset):
    '''
    Custom PyTorch Dataset for loading and processing audio files.
    '''
    def __init__(self, file_paths, labels, augmentations=None):
        self.file_paths = file_paths
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]
        audio = ai.model.open_audio_file(filepath)

        # Apply Augmentations
        if self.augmentations:
            try:
                audio = self.augmentations(
                        samples=audio,
                        sample_rate=ai.model.SAMPLE_RATE)
            except Exception:
                pass

        spectrogram = ai.model.audio_to_spectrogram(audio)
        return spectrogram, torch.tensor(label, dtype=torch.long)


def train_all():
    '''
    Main entry point for the training script.

    Train all configured models using tagged audio clips.
    '''
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])
    models_dir = workspace / 'models'

    for model_name in options['inspect_models']:
        logging.debug('Begin training: %s', model_name)
        try:
            num_classes = train_model(model_name, options)
            logging.info('Finished training %s', model_name)
            ai.model.convert(
                models_dir / f'{model_name}.pth',
                models_dir / f'{model_name}.onnx',
                num_classes
            )
            logging.info('MODEL PREPARED: %s', model_name)
        except KeyboardInterrupt:
            logging.info('Received Ctrl+C; Training stopped')
        except Exception as e:
            logging.error('Failed to train %s: %s', model_name, e)
            raise e


def train_model(model_name, options):
    '''
    Conducts the full training and validation process for a model.

    Uses training data from workspace/tags/<model_name>/<category>,
    produces <workspace>/models/<model_name>.pth,
    returns total number of identified <categories>.
    '''
    workspace = pathlib.Path(options['workspace'])
    models_dir = workspace / 'models'
    data_dir = workspace / 'tags' / model_name

    # Verify tags/<model_name>/<data> exists
    if not data_dir.exists():
        raise OSError(f'Data directory not found for model: {data_dir}')
    if not (data_dir / 'empty').exists():
        raise OSError(f'No "null" data found: {data_dir}/empty')

    # Identify all classes from existing folder structure
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(class_folders) < 2:
        raise ValueError('Need at least 2 class folders.')

    classes = [d.name for d in class_folders]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    logging.info('Detected categories for %s: %s', model_name, classes)

    models_dir.mkdir(parents=True, exist_ok=True)

    # Save Labels Map
    with open(models_dir / f'{model_name}_labels.json', 'w') as fh:
        json.dump(classes, fh)

    all_files = []
    all_labels = []
    class_counts = []

    for cls_name in classes:
        folder = data_dir / cls_name
        files = list(folder.glob('*.dat'))
        count = len(files)
        logging.debug('Class "%s": %d samples', cls_name, count)
        class_counts.append(count)
        all_files.extend(files)
        all_labels.extend([class_to_idx[cls_name]] * count)

    if not all_files:
        raise OSError('No .dat files found.')

    # Weights
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(classes) * c) for c in class_counts]
    logging.debug('Class Weights: %s', class_weights)

    train_files, val_files, train_labels, val_labels = \
        sklearn.model_selection.train_test_split(
                all_files, all_labels, test_size=0.2,
                random_state=42, stratify=all_labels)

    # Augmentations
    train_dataset = AudioDataset(
            train_files,
            train_labels,
            augmentations=audiomentations.Compose([
                audiomentations.AddGaussianNoise(
                    min_amplitude=0.001, max_amplitude=0.02, p=0.5),
                audiomentations.TimeStretch(
                    min_rate=0.8, max_rate=1.2, p=0.5),
                audiomentations.PitchShift(
                    min_semitones=-4, max_semitones=4, p=0.5),
                audiomentations.Shift(
                    min_shift=-0.2, max_shift=0.2, p=0.5),
            ]))

    val_dataset = AudioDataset(val_files, val_labels, augmentations=None)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=options['train_batch_size'],
            shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=options['train_batch_size'],
            shuffle=False)

    num_classes = len(classes)
    dev = ai.model.CUDA_CPU
    model = ai.model.NoiseDetector(num_classes=num_classes).to(dev)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(dev)

    # Label Smoothing + Weights
    criterion = torch.nn.CrossEntropyLoss(
            weight=weights_tensor, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options['train_learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    epochs_worse = 0

    for epoch in range(options['train_epochs']):
        model.train()
        train_loss, train_correct = 0, 0

        for inputs, labels in tqdm.tqdm(
                train_loader, desc=f'Epoch {epoch} [Train]',
                leave=False, file=sys.stdout):
            inputs, labels = inputs.to(dev), labels.to(dev)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(
                    val_loader, desc=f'Epoch {epoch} [Check]',
                    leave=False, file=sys.stdout):
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / len(train_dataset)
        val_accuracy = 100 * val_correct / len(val_dataset)

        logging.debug(
            '#%d: Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%',
            epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            logging.info(
                    'Model #%d improved (Loss: %.4f); Saving',
                    epoch, avg_val_loss)
            ai.model.save(model, models_dir / f'{model_name}.pth')
            best_val_loss = avg_val_loss
            epochs_worse = 0
        else:
            epochs_worse += 1

        if epochs_worse >= options['train_patience']:
            logging.info('Training patience exhausted.')
            return num_classes

    logging.info('Training epochs exhausted; check quality of audio samples!')
    return num_classes


if __name__ == '__main__':
    train_all()
