"""
To build the data for training from the mat files, and split.
"""

import os
import glob
import csv

import pandas as pd
from sklearn.model_selection import RepeatedKFold
from scipy.io import loadmat

import numpy as np

from HuashanHand_new_copy.Old.config import config, select_brain_region

EVA_names = config['EVA_names']
PIDwidth = config['PIDwidth']
ROINum = config['ROINum']

currentFold = config['currentFold']
totalFold = config['totalFold']


def get_pid(input_file):
    """
    the pid is in three digits: e.g. 001
    :param input_file: file name
    :return: PID with 3 digits
    """
    root, _ = os.path.splitext(input_file)
    Bname = os.path.basename(root)
    itemPID = Bname.split('_')[-1]
    itemPID = itemPID[-3:]

    return itemPID


def pid_from_excel(item):
    currentName = str(item)
    currentName = currentName.zfill(PIDwidth)

    return currentName


def read_mat(input_list, excel_input, output_path, save=False):
    GTdf = pd.read_excel(excel_input)
    PID_list = GTdf['PID'].to_list()
    modified_PID_List = [pid_from_excel(item) for item in PID_list]
    Injury_side_list = GTdf['Injury_side'].to_list()

    Injury_side_dict = dict()
    NN = len(modified_PID_List)
    for i in range(NN):
        Injury_side_dict[modified_PID_List[i]] = Injury_side_list[i]

    SignalDict = dict()
    i = 0
    for item in input_list:
        itemPID = get_pid(item)
        key_list = list(Injury_side_dict.keys())
        if itemPID in key_list:
            Injury_side = Injury_side_dict[itemPID]
            selected_order = select_brain_region[Injury_side]
            itemSignal = loadmat(item)['ROISignals']
            usefulSignal = itemSignal[:, selected_order]
            SignalDict[str(itemPID)] = usefulSignal
            if i == 0:
                outputShape = usefulSignal.shape
            i = i + 1

    return SignalDict, outputShape


def main():
    ...


if __name__ == "__main__":
    main()
