import os
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
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        PID = str(self.PIDs[idx])
        file_name = os.path.join(self.data_dir, f'ROISignals_sub10{PID}.mat')

        signal = loadmat(file_name)
        label = self.labels[idx]
        sample = {'img': signal, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    input_dir = os.getcwd()
    label_file_name = 'HuashanHand_new_copy/Data/ROISignals_FunRawRWSDCF/Patient_Information_2019_09_29.xlsx'
    data_dir = os.path.abspath('./HuashanHand_new_copy/Data/ROISignals_FunRawRWSDCF')

    dataset = CustomDataSet(input_dir, label_file_name, data_dir, transform=None)
