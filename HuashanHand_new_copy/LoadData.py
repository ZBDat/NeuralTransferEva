import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from scipy.io import loadmat


class CustomDataSet(Dataset):

    def __init__(self, input_dir, label_file_name, data_dir, transform=None):
        self.input_dir = input_dir
        label_file = os.path.join(self.input_dir, label_file_name)

        if not os.path.exists(label_file):
            raise ValueError("File path does not exist!")

        _, ext = os.path.splitext(label_file)
        if ext not in ['.xls', '.xlsx']:
            raise ValueError("The input file should be an Excel file with extension .xls or .xlsx!")

        else:
            self.label_file = os.path.join(self.input_dir, label_file_name)

        self.label_info = pd.read_excel(self.label_file)
        self.PIDs = self.label_info['PID'].tolist()
        self.labels = self.label_info['Kangfu'].tolist()

        self.data_dir = data_dir

        for PID in self.PIDs:
            file_name = os.path.join(self.data_dir, f'ROISignals_sub10{PID}.mat')
            if not os.path.exists(file_name):
                del self.labels[self.PIDs.index(PID)]
                self.PIDs.remove(PID)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        PID = str(self.PIDs[idx]).zfill(3)
        file_name = os.path.join(self.data_dir, f'ROISignals_sub10{PID}.mat')

        # some PID has no corresponding file

        signal = loadmat(file_name)
        signal = signal['ROISignals']
        signal = torch.from_numpy(signal).unsqueeze(0)

        label = self.labels[idx]
        sample = {'signal': signal, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


def load_data(input_dir: str, label_file_name, data_dir):
    dataset = CustomDataSet(input_dir, label_file_name, data_dir, transform=None)
    print('Data Loaded with total number of {} samples.'.format(len(dataset)))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print('Split into train: {} and test: {}.'.format(train_size, test_size))

    train_loader = DataLoader(train_set, batch_size=5, drop_last=False, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=5, drop_last=False, shuffle=False, num_workers=2)

    return train_loader, test_loader


if __name__ == "__main__":
    input_dir = os.getcwd()
    label_file_name = os.path.join('Data', 'ROISignals_FunRawRWSDCF', 'Patient_Information_2019_09_29.xlsx')
    data_dir = os.path.join(os.getcwd(), 'Data', 'ROISignals_FunRawRWSDCF')

    train_loader, test_loader = load_data(input_dir, label_file_name, data_dir)

    for i, data in enumerate(test_loader):
        signal = data['signal']
        label = data['label']
        print('num:{}| signal:{}| label:{}'.format(i, signal.size(), label))
