#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:10:29 2019

@author: zhaoyu
"""
import os
import glob
import csv
import random
import pandas as pd
import numpy as np 
from config import config, select_brain_region
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold

EVA_names = config['EVA_names']
PIDwidth = config['PIDwidth']
# ROINum = config['ROINum']

currentFold = config['currentFold']
totalFold = config['totalFold'] 

#%%
def get_PID_from_mat_file(input_file):
    
    root, _ = os.path.splitext(input_file)
    Bname = os.path.basename(root)
    itemPID = Bname.split('_')[-1]
    itemPID = itemPID[-3:]

    return itemPID    

def get_PID_from_excel_file(item):
    currentName = str(item)
    currentName = currentName.zfill(PIDwidth)

    return currentName

def splitData(X_data,Y_data, testNumPer):
    SampleNum = np.shape(X_data)[0]
    currentList = range(SampleNum)
    random.shuffle(currentList)
    testNum = int(testNumPer*SampleNum)
    testID = currentList[:testNum]
    trainingID = currentList[testNum:]
    X_train = X_data[trainingID,:]
    Y_train = Y_data[trainingID,:]
    X_test = X_data[testID,:]
    Y_test = Y_data[testID,:]
    return X_train, Y_train, X_test, Y_test

def split_data_to_train_val(X_data, Aux_data, Y_data, output_folder, n_splits, random_state, shuffle=True):

    Stratifiedkf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    Fold_ID = 1
    for train_index, test_index in Stratifiedkf.split(X_data, Y_data):
        print('train:{}'.format(train_index))
        print('test:{}'.format(test_index))
        X_train, X_test = X_data[train_index], X_data[test_index]
        aux_train, aux_test = Aux_data[train_index], Aux_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]

        np.save(os.path.join(outputfolder, 'x_train_{}.npy'.format(Fold_ID)), X_train)
        np.save(os.path.join(outputfolder, 'aux_train_{}.npy'.format(Fold_ID)), aux_train)
        np.save(os.path.join(outputfolder, 'y_train_{}.npy'.format(Fold_ID)), y_train)
        np.save(os.path.join(outputfolder, 'x_test_{}.npy'.format(Fold_ID)), X_test)
        np.save(os.path.join(outputfolder, 'aux_test_{}.npy'.format(Fold_ID)), aux_test)
        np.save(os.path.join(outputfolder, 'y_test_{}.npy'.format(Fold_ID)), y_test)     
        
        print('finish fold: {}'.format(Fold_ID))
        Fold_ID += 1

def readMatToNpy(inputList, outputPath, ifSave=False):
    SignalDict = dict()
    i = 0
    for item in inputList:        
        itemPID = get_PID_from_mat_file(item)
        itemSignal = loadmat(item)['ROISignals']
        usefulSignal = itemSignal[:,0:ROINum]
        SignalDict[str(itemPID)]=usefulSignal
        if i == 0:
            outputShape = usefulSignal.shape
        i = i+1

    if ifSave == True:
        outputdir = os.path.join(outputPath,'PatientRoiSignal.csv')
        with open(outputdir,'wb') as f:
            w = csv.writer(f)
            w.writerows(SignalDict.items())
    return SignalDict, outputShape

def readExcelToNpy(inputPath, outputPath, ifSave=False):
    if not isinstance(inputPath, str):
        raise ValueError("The GroundTruthFilePath should be the path of an Excel file storing the Ground Truth information")

    _, ext = os.path.splitext(inputPath)

    if ext not in ['.xls','.xlsx']:
        raise ValueError("The input file should be an Excel file with extension .xls or .xlsx")

    GTdf = pd.read_excel(inputPath)

    PID_list = GTdf['PID'].to_list()
    label_list = GTdf['score'].to_list()

    NN = len(PID_list)

    GTdict = dict()
    for i in range(NN):
        currentName = get_PID_from_excel_file(PID_list[i])

        currentEva = str(label_list[i])
        if currentEva in EVA_names:
            currentEva = EVA_names.index(currentEva)
        else:
            raise ValueError('The disease of Patient:' + currentName + ' is not in the disease list.' )
        GTdict[currentName]=currentEva

    if ifSave == True:
        outputdir = os.path.join(outputPath,'PatientLabel.csv')
        with open(outputdir,'wb') as f:
            w = csv.writer(f)
            w.writerows(GTdict.items())
    return GTdict

def readExcelToNpy_v1(inputPath, outputPath, FMS_UE_threshold, ifSave=False):
    if not isinstance(inputPath, str):
        raise ValueError("The GroundTruthFilePath should be the path of an Excel file storing the Ground Truth information")

    _, ext = os.path.splitext(inputPath)

    if ext not in ['.xls','.xlsx']:
        raise ValueError("The input file should be an Excel file with extension .xls or .xlsx")

    GTdf = pd.read_excel(inputPath)

    PID_list = GTdf['PID'].to_list()
    label_list = GTdf['FMS-UE_change'].to_list()
    Kangfu_list = GTdf['Kangfu'].to_list()

    NN = len(PID_list)

    GTdict = dict()
    aux_info_dict = dict() 
    output_GT_list = list()
    output_PID_list = list()
       
    for i in range(NN):
        currentName = get_PID_from_excel_file(PID_list[i])
        currentEva = int(label_list[i] >= FMS_UE_threshold)
        GTdict[currentName]=currentEva
        temp_df = GTdf.iloc[i,:]
        temp_df = temp_df.iloc[2:13]
        aux_info_dict[currentName]=temp_df.to_numpy()
        output_GT_list.append(currentEva)
        output_PID_list.append(currentName)

    if ifSave == True:
        outputdir = os.path.join(outputPath,'PatientLabel.csv')
        df = pd.DataFrame({'PID': output_PID_list, 'GroundTruth': output_GT_list, 'Kangfu': Kangfu_list})
        df.to_csv(outputdir)
    return GTdict, aux_info_dict

def readMatToNpy_v1(inputList, ExcelInput, outputPath, ifSave=False):
    GTdf = pd.read_excel(ExcelInput)
    PID_list = GTdf['PID'].to_list()
    modified_PID_List = [get_PID_from_excel_file(item) for item in PID_list]
    Injury_side_list = GTdf['Injury_side'].to_list()

    Injury_side_dict=dict()
    NN = len(modified_PID_List)
    for i in range(NN):
        Injury_side_dict[modified_PID_List[i]] = Injury_side_list[i]
    
    SignalDict = dict()
    i = 0
    for item in inputList:        
        itemPID = get_PID_from_mat_file(item)
        key_list = list(Injury_side_dict.keys())
        if itemPID in key_list:
            Injury_side = Injury_side_dict[itemPID]        
            selected_order = select_brain_region[Injury_side]
            itemSignal = loadmat(item)['ROISignals']
            usefulSignal = itemSignal[:,selected_order]
            SignalDict[str(itemPID)]=usefulSignal
            if i == 0:
                outputShape = usefulSignal.shape
            i = i+1

    return SignalDict, outputShape

def main(inputfolder, outputfolder, config):

    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    inputList = glob.glob(inputfolder+'/ROISignals_sub*.mat')
    ExcelInput = os.path.join(inputfolder,'Patient_Information_2019_09_29.xlsx')
    SignalDict, outputShape = readMatToNpy_v1(inputList, ExcelInput, outputfolder, ifSave=False)
    inputPath = os.path.join(inputfolder,'Patient_Information_2019_09_29.xlsx')
    labelDict, aux_info_dict = readExcelToNpy_v1(inputPath, outputfolder, FMS_UE_threshold=15, ifSave=True)
    
    # prepare trainging data and label
    sorted_keys = sorted(list(set(labelDict.keys()).intersection(set(SignalDict.keys()))), reverse=False)
    NN = len(sorted_keys)
    size = (NN,)+outputShape
    ImgArray = np.zeros(size)
    AuxInfoArray = np.zeros((NN, 11))
    LabelArray = np.zeros((NN,1))

    ii=0
    for currentKey in sorted_keys:
        print('Current Key is :{}'.format(currentKey))
        ImgArray[ii] =  SignalDict[currentKey]
        AuxInfoArray[ii] = aux_info_dict[currentKey]
        LabelArray[ii] = labelDict[currentKey]
        ii += 1        
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    split_data_to_train_val(ImgArray, AuxInfoArray, LabelArray, outputfolder, n_splits = totalFold, random_state = config['random_state'], shuffle=True)

#%%
if __name__=='__main__':
    inputfolder = os.path.abspath('../Data/ROISignals_FunRawRWSDCF')
    outputfolder = os.path.abspath('./dataStore')
    main(inputfolder, outputfolder, config)