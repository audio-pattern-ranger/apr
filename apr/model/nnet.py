'''
PyTorch Model

This module defines two neural network architectures using PyTorch:
a simple fully connected network (Net) and a convolutional network (M5)
'''
import logging
import pathlib
import glob
import torch
import torchaudio

# APR
import apr.config

# Convenient aliases
from torch import nn
import torch.nn.functional as F


class M5(nn.Module):
    '''
    Convolutional neural network with multiple convolutional and pooling layers
    '''
    def __init__(self, n_input=1, n_output=2, stride=16, n_channel=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=49, stride=16)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=49)
        self.bn2 = nn.BatchNorm1d(2 * n_channel)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=7)
        self.bn3 = nn.BatchNorm1d(4 * n_channel)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(4 * n_channel, 2 * n_channel, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(2 * n_channel, 1 * n_channel, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(1 * n_channel)
        self.fc1 = nn.Linear(1 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, models, transform=None, n_input=256):
        self.audio_list = glob.glob(f'{root_dir}/*/*.wav')
        self.transform = transform
        self.n_input = n_input
        self.mean, self.std = self._compute_mean()
        self.label2index = {m: i for i, m in enumerate(models)}

    def _compute_mean(self):
        meanstd_file = pathlib.Path(apr.config.get('workspace')) / '_mean.pth'
        if meanstd_file.exists():
            meanstd = torch.load(meanstd_file)
        else:
            logging.debug('computing _mean.pth')
            mean = torch.zeros(self.n_input)
            std = torch.zeros(self.n_input)
            cnt = 0
            for path in self.audio_list:
                cnt += 1
                logging.debug(f' {cnt} | {len(self.audio_list)}')

                wv, _ = torchaudio.load(path)
                if wv.shape[0] > 1:
                    wv = torch.mean(wv, axis=0, keepdim=True)

                for tr in self.transform:
                    wv = tr(wv)
                mean += wv.mean(1)
                std += wv.std(1)

            mean /= len(self.audio_list)
            std /= len(self.audio_list)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)

        return meanstd['mean'], meanstd['std']

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        path = self.audio_list[idx]
        class_name = path.split('/')[-2]
        label = self.label2index[class_name]

        wv, sr = torchaudio.load(path)
        if wv.shape[0] > 1:
            wv = torch.mean(wv, axis=0, keepdim=True)
        audio_feature = wv
        if self.transform:
            for tr in self.transform:
                audio_feature = tr(audio_feature)

        return audio_feature, torch.tensor(label)
