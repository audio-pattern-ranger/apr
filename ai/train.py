'''
Disturbance Tracker - Model Trainer

This script trains a PyTorch-based neural network to classify audio clips.
It handles data loading, augmentation, training, validation, and reporting.
'''
import logging
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# DTrack
import ai.model
import ai.options

class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing audio files.
    """
    def __init__(self, file_paths, labels, augmentations=None):
        """
        Initializes the dataset.

        Args:
            file_paths (list): List of paths to the audio files.
            labels (list): Corresponding list of labels (0 or 1).
            augmentations (callable, optional): Audiomentations Compose object.
        """
        self.file_paths = file_paths
        self.labels = labels
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
            audio = self.augmentations(samples=audio, sample_rate=ai.model.SAMPLE_RATE)

        # Convert audio to a spectrogram
        spectrogram = ai.model.audio_to_spectrogram(audio)

        return spectrogram, torch.tensor(label, dtype=torch.float32)


def main():
    """
    Main entry point for the training script.
    """
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])
    tags_dir = workspace / 'tags'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    for model_name in options['inspect_models']:
        logging.info(f"--- Starting training for model: {model_name} ---")

        match_dir = tags_dir / model_name
        empty_dir = tags_dir / 'empty'

        if not match_dir.exists() or not empty_dir.exists():
            logging.error(f"Missing data directories for model '{model_name}'. Skipping.")
            continue

        # Train and validate the model
        model = train(model_name, match_dir, empty_dir, device, options)

        # Save the final trained model
        if model:
            output_path = workspace / f'{model_name}.pth'
            ai.model.save_model(model, output_path)
            logging.info(f"Successfully saved final model to {output_path}")

        logging.info(f"--- Finished training for model: {model_name} ---")


def train(model_name, match_dir, empty_dir, device, options):
    """
    Conducts the full training and validation process for a model.

    Args:
        model_name (str): The name of the model being trained.
        match_dir (pathlib.Path): Directory with positive audio samples.
        empty_dir (pathlib.Path): Directory with negative audio samples.
        device (torch.device): The device to train on.
        options (dict): Configuration dictionary.

    Returns:
        ai.model.NoiseDetector: The trained model instance.
    """
    # 1. Prepare Dataset
    logging.info("Loading and preparing dataset...")
    match_files = list(match_dir.glob('*.dat'))
    empty_files = list(empty_dir.glob('*.dat'))

    if not match_files or not empty_files:
        logging.error("Not enough data to train. Both match and empty folders need samples.")
        return None

    all_files = match_files + empty_files
    # labels = * len(match_files) + * len(empty_files)
    labels = [1] * len(match_files) + [0] * len(empty_files)

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logging.info(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    # 2. Define Augmentations and Create DataLoaders
    train_augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    ])

    train_dataset = AudioDataset(train_files, train_labels, augmentations=train_augmentations)
    val_dataset = AudioDataset(val_files, val_labels, augmentations=None)

    train_loader = DataLoader(train_dataset, batch_size=options['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=options['train_batch_size'], shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    logging.info(f"Initializing model: {ai.model.MODEL_NAME}")
    model = ai.model.NoiseDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=options['train_learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # 4. Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_path = pathlib.Path(f"best_{model_name}_model.pth")

    for epoch in range(options['train_epochs']):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()

        # --- Logging and History ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / len(train_dataset)
        val_accuracy = 100 * val_correct / len(val_dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        logging.info(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%"
        )

        scheduler.step(avg_val_loss)

        # --- Early Stopping and Model Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ai.model.save_model(model, best_model_path)
            epochs_no_improve = 0
            logging.debug(f"Validation loss improved. Saving best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.debug(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= options['train_patience']:
            logging.info(f"Early stopping triggered after {options['train_patience']} epochs.")
            break

    # 5. Final Evaluation and Reporting
    logging.info("Training complete. Generating final report using the best model...")
    model = ai.model.load_model(best_model_path, device)
    generate_reports(model, val_loader, device, model_name)

    return model

def generate_reports(model, val_loader, device, model_name):
    """
    Generates and saves classification report, confusion matrix, and training curves.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=['empty', model_name])
    logging.info(f"\nClassification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['empty', model_name])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    logging.info("Confusion matrix saved to confusion_matrix.png")
    plt.close()

if __name__ == '__main__':
    main()