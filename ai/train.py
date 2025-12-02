'''
Disturbance Tracker - Model Trainer

Uses directory structure: tags/<model_name>/<class>/<audio_segment>.dat
'''
import logging
import pathlib
import json
import sys

# 3rd-Party
import audiomentations
import numpy as np
import sklearn.metrics
import sklearn.model_selection
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
        self.file_paths = file_paths
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        # Load (and prepare) audio
        audio = ai.model.open_audio_file(filepath)

        # Apply Augmentations
        if self.augmentations:
            audio = self.augmentations(
                    samples=audio,
                    sample_rate=ai.model.SAMPLE_RATE)

        # Return Label as LongTensor for CrossEntropyLoss
        spectrogram = ai.model.audio_to_spectrogram(audio)
        return spectrogram, torch.tensor(
                label, dtype=torch.long, device=ai.model.CUDA_CPU)


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

        # Train (and get class count)
        try:
            num_classes = train_model(model_name, options)
        except KeyboardInterrupt:
            logging.info('Received Ctrl+C; Training stopped')

        # Run validation
        build_report(model_name, options)

        # Export to ONNX format
        ai.model.convert(
            models_dir / f'{model_name}.pth',
            models_dir / f'{model_name}.onnx',
            num_classes)

        logging.info('Finished training: %s', model_name)


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

    # 1. Identify all classes from existing folder structure
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(class_folders) < 2:
        raise ValueError('Need at least 2 class folders.')

    # Create mapping: Name -> Index
    classes = [d.name for d in class_folders]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    logging.info('Detected categories for %s: %s', model_name, classes)

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    # Labels Map: Gather files and Calculate Counts for Balancing
    with open(
            models_dir / f'{model_name}.labels', 'w',
            encoding='utf-8') as fh:
        json.dump(classes, fh)

    # Gather files and Calculate Counts for Balancing
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

    # Calculate Class Weights for Imbalance
    # Formula: Total / (NumClasses * ClassCount)
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(classes) * c) for c in class_counts]
    logging.debug('Class Weights: %s', class_weights)

    # Weights (Stratified Split)
    train_files, val_files, train_labels, val_labels = \
        sklearn.model_selection.train_test_split(
                all_files, all_labels, test_size=0.2,
                random_state=42, stratify=all_labels)

    # 2. Augmentations and Loaders
    train_dataset = AudioDataset(
            train_files,
            train_labels,
            augmentations=audiomentations.Compose([
                audiomentations.AddGaussianNoise(
                    min_amplitude=0.001, max_amplitude=0.02, p=0.2),
                audiomentations.TimeStretch(
                    min_rate=0.95, max_rate=1.05,
                    leave_length_unchanged=True, p=0.2),
                audiomentations.Shift(
                    min_shift=-0.2, max_shift=0.2, p=0.2),
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

    # 3. Initialize Model
    num_classes = len(classes)
    dev = ai.model.CUDA_CPU
    model = ai.model.NoiseDetector(num_classes=num_classes).to(dev)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(dev)

    # Label Smoothing adds random error to prevent over-fitting via human error
    criterion = torch.nn.CrossEntropyLoss(
            weight=weights_tensor, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options['train_learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=4)

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_worse = 0
    for epoch in range(options['train_epochs']):

        # Training Phase
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm.tqdm(
                train_loader, desc=f'Epoch {epoch} [Train]',
                leave=False, file=sys.stdout):
            inputs = inputs.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy: Max argument of Softmax
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(
                    val_loader, desc=f'Epoch {epoch} [Check]',
                    leave=False, file=sys.stdout):
                inputs = inputs.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        # Log Results (Using logging.info to show in file)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / len(train_dataset)
        val_accuracy = 100 * val_correct / len(val_dataset)
        logging.debug(
                '#%d: Train Loss: %.4f, Acc: %.2f%% |'
                ' Val Loss: %.4f, Acc: %.2f%%',
                epoch, avg_train_loss, train_accuracy,
                avg_val_loss, val_accuracy)

        # Adjust learning rate if plateau was hit
        scheduler.step(avg_val_loss)

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

    logging.info('Training epochs exhausted.')
    return num_classes


def build_report(model_name, options):
    '''
    Load the trained model, run inference on validation data,
    produce a confusion matrix, and save report.
    '''
    workspace = pathlib.Path(options['workspace'])
    models_dir = workspace / 'models'
    data_dir = workspace / 'tags' / model_name

    # Load labels
    with open(models_dir / f'{model_name}.labels', 'r') as f:
        classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load model
    model = ai.model.load(models_dir / f'{model_name}.pth', len(classes))

    # Gather validation files and labels
    all_files = []
    all_labels = []
    for cls in classes:
        files = list((data_dir / cls).glob('*.dat'))
        all_files.extend(files)
        all_labels.extend([class_to_idx[cls]] * len(files))
    if not all_files:
        print(f'No validation data found for {model_name}.')
        return

    # Prepare dataset
    val_dataset = AudioDataset(all_files, all_labels, augmentations=None)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=options['train_batch_size'],
            shuffle=False)

    # Collect predictions
    preds = []
    truths = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(ai.model.CUDA_CPU)
            labels = labels.to(ai.model.CUDA_CPU)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = sklearn.metrics.confusion_matrix(
            truths, preds, labels=range(len(classes)))

    # Save report showing model accuracy
    with open(
            models_dir / f'{model_name}_report.txt',
            'w', encoding='utf-8') as fh:
        fh.write(f'Final report for model: {model_name}\n')

        # Save confusion matrix
        fh.write('\n= CONFUSION MATRIX =\n\n')
        fh.write('Predicted: \\ Actual:  ')
        fh.write(''.join([f'{c:<14}' for c in classes]) + '\n')
        for i, row in enumerate(cm):
            fh.write(f'  {classes[i]:<20}')
            fh.write(''.join([f'{val:<14}' for val in row]) + '\n')

        # Find misclassified files
        fh.write('\n= MISCLASSIFIED FILES =\n\n')
        for i, (file, e, p) in enumerate(zip(all_files, truths, preds)):
            if e != p:
                fh.write('  {}  Expected: {}  Predicted: {}\n'.format(
                    file.name, classes[e], classes[p]))


if __name__ == '__main__':
    train_all()
