'''
Train a Machine Learning Model
'''
import logging
import pathlib
import shutil
import torch
import torchaudio

# APR
import apr.config
import apr.model.nnet


class AudioClassifier:
    '''
    TODO
    '''
    def __init__(self):
        # Check for cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.debug(f'Backend: {self.device}')

        # Data locations
        self.workspace = pathlib.Path(apr.config.get('workspace'))
        self.model_path = self.workspace / 'model.pth'
        self.training_data = self.workspace / 'train'
        self.testing_data = self.workspace / 'test'
        self.sample_rate = None
        self._primed = False
        self._loaders = {}

        # Tuning options
        self.batch_size = 1
        self.momentum = 0.9
        self.learning_rate = 0.001
        self.target_accuracy = apr.config.get('target_accuracy')

        # Models (search labels)
        self.models = apr.config.get('models')
        self.label2index = {m: i for i, m in enumerate(self.models)}
        self.index2label = {i: m for i, m in enumerate(self.models)}

        # ML Model - Assumes single channel audio
        self.network = apr.model.nnet.M5(n_input=1).to(self.device)
        if self.model_path.exists():
            logging.debug(f'Loading previous state from {self.model_path}')
            self.network.load_state_dict(torch.load(self.model_path))
            self._primed = True
        logging.debug(f'Number of params: {count_parameters(self.network)}')

        # Training state
        self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10, 30], gamma=0.1)

        # Load a sample clip to build transformation
        self._load_sample()

    def _load_sample(self):
        '''
        Sample the first available input clip
        '''
        sample_clip = self.workspace / 'model.wav'

        # Grab the first available clip if sample does not exist
        if not sample_clip.exists():
            demo_clip = next((self.training_data / 'nomatch').glob('*.wav'))
            shutil.copy(demo_clip, sample_clip)
            logging.debug(f'Generated model.wav sample from {demo_clip}')

        # Load sample for the transformation engine
        logging.debug(f'Generated model.wav sample from {sample_clip}')
        self.waveform, self.sample_rate = self.load_audio(sample_clip)

    def load_audio(self, file_path):
        '''
        Load audio and transform to mono channel
        '''
        wv, sample_rate = torchaudio.load(file_path)
        # Convert multiple channels to mono
        if wv.shape[0] > 1:
            wv = wv.mean(dim=0, keepdim=True)
        transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000)
        transformed = transform(wv)
        return (transformed, sample_rate)

    def get_loader(self, data):
        '''
        Returns a data loader after ensuring it is loaded
        '''
        # Return loader if already loaded
        if data in self._loaders:
            return self._loaders[data]

        # Load specified loader
        path = getattr(self, data)
        dataset = apr.model.nnet.NoiseDataset(
                root_dir=path, models=self.models,
                transform=[torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, new_freq=16000)])
        loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size,
                shuffle=True, collate_fn=collate_fn)

        # Save and return loader
        self._loaders[data] = loader
        return loader

    def training_loop(self):
        '''
        Continue testing new models until target_accuracy is met
        '''
        # Track best iteration
        best_i = 0
        # Use defaults if no model was loaded
        if not self._primed:
            best = dict.fromkeys(self.models, 0)
            avg_best = 0
        else:
            best = self.evaluate()
            avg_best = sum(best.values()) / len(best)
            logging.info(f'Initial accuracy[0] is {avg_best}')

        # Continue training until desired threshold is met
        iteration = 0
        last_accuracy = [0, 0]  # [accuracy, count]
        while avg_best < float(self.target_accuracy):
            iteration += 1

            # Train a new model
            logging.info(f'Training iteration {iteration}')
            self.train_once()

            # Check accuracy of new model
            logging.debug(f'Testing accuracy of iteration {iteration}')
            accuracy = self.evaluate()
            logging.info('Overall accuracy[{i}] is {a}'.format(
                i=iteration, a=sum(accuracy.values()) / len(best)))

            # Save the model if accuracy improves
            logging.debug('Old Accuracy: {old}  New Accuracy: {new}'.format(
                old=sum(best.values()), new=sum(accuracy.values())))
            if sum(accuracy.values()) > sum(best.values()):
                logging.info('Accuracy increased; keeping new model')
                torch.save(self.network.state_dict(), self.model_path)
                best = accuracy.copy()
                best_i = iteration
                avg_best = sum(best.values()) / len(best)
            else:
                logging.info(f'Accuracy worse than #{best_i}; discarding new')

            # Check for an infinite loop (if target_accuracy cannot be met)
            if sum(accuracy.values()) != last_accuracy[0]:
                last_accuracy = [sum(accuracy.values()), 0]
            else:
                last_accuracy[1] += 1
                if last_accuracy[1] >= 10:
                    logging.warning('New accuracy unchanged for 10 iterations')
                    break

        logging.info(f'TRAINING COMPLETE :: Final Accuracy: {avg_best}')

    def train_once(self):
        '''
        Train a model using collected data
        '''
        criterion = torch.nn.CrossEntropyLoss()
        running_loss, correct, total = 0.0, 0, 0
        for i, (inputs, labels) in enumerate(self.get_loader('training_data')):

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.network(inputs.to(self.device)).squeeze(1)
            loss = criterion(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.to(self.device).size(0)
            correct += (predicted == labels.to(self.device)).sum().item()

            # Print statistics every 20 mini-batches
            if i % 2000 == 1999:
                logging.debug('#{i:5d} Loss: {l:.3f} Accuracy: {a:.1f}'.format(
                    i=i, l=running_loss / 2000, a=100 * correct // total))
                # Reset tally
                running_loss, correct, total = 0.0, 0, 0

        self.scheduler.step()

    def evaluate(self):
        correct_pred = dict.fromkeys(self.models, 0)
        total_pred = dict.fromkeys(self.models, 0)

        with torch.no_grad():
            for inputs, labels in self.get_loader('testing_data'):
                outputs = self.network(inputs.to(self.device)).squeeze(1)
                _, predictions = torch.max(outputs, 1)

                # Collect correct predictions for each class
                for label, prediction in zip(
                        labels.to(self.device), predictions):
                    if label == prediction:
                        correct_pred[self.index2label[label.item()]] += 1
                    total_pred[self.index2label[label.item()]] += 1

        # Calculate accuracy for each class
        accuracy = {}
        for cls, correct_count in correct_pred.items():
            accuracy[cls] = 100 * float(correct_count) / total_pred[cls]
            logging.debug(f'Accuracy for {cls:5s} is {accuracy[cls]:.1f}%')

        return accuracy


def collate_fn(batch):
    '''
    Aggregates a list of data samples into a batched tensor
    suitable for input into a neural network
    '''
    # A data tuple has the form: waveform, label
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def pad_sequence(batch):
    '''
    Pad all tensors in a batch to the same length using zeros
    '''
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def count_parameters(model):
    '''
    Return the number of trainable parameters in a PyTorch model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
