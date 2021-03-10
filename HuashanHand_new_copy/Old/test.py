#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:10:29 2019

@author: zhaoyu
"""
from __future__ import division
import os
import SimpleITK as sitk
import glob
import numpy as np
import pandas as pd

#disease_names=['Y','N']
#inputPath = '/Users/zhaoyu/Projects/data/HuashanHandSample/labelInformation.xlsx'
#if not isinstance(inputPath, basestring):
#    raise ValueError("The GroundTruthFilePath should be the path of an Excel file storing the Ground Truth information")
#
#_, ext = os.path.splitext(inputPath)
#
#if ext not in ['.xls','.xlsx']:
#    raise ValueError("The input file should be an Excel file with extension .xls or .xlsx")
#
#GTdata = pd.read_excel(inputPath, index_col=None, header=None)
#
#if GTdata[0][0]==u'PID' and GTdata[6][0]==u'score':
#    GTarray = GTdata.values
#    NN,col = GTarray.shape
#    # remove unnecessary rows and columns
#    NewGTarray =  GTarray[1:NN,[0,6]]
#    GTdict = dict()
#    # modify the PED ID the formate should be patientID_petID
#    for i in range(NN-1):
#        currentName = str(NewGTarray[i,0])
#        currentEva = str(NewGTarray[i,1])
#        if currentEva in disease_names:
#            currentEva = disease_names.index(currentEva)
#        else:
#            raise ValueError('The disease of Patient:' + currentName + ' is not in the disease list.' )
#        GTdict[currentName]=currentEva

#%%
# filepath = '/media/data/yuzhao/data_restore/PSMA_FirstTime_original/Bern/BernCO-R'
# filelist = glob.glob(filepath + '/*.nii.gz')
# for item in filelist:
#     root, ext = os.path.splitext(item)
#     outputFile = root + '.nii'
#     image = sitk.ReadImage(item)
#     sitk.WriteImage(image,outputFile)
#     os.remove(item) 
    
#%%
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

inputfolder = os.path.abspath('../Data/ROISignals_FunRawRWSDCF')
ExcelInput = os.path.join(inputfolder,'Patient_Information_2019_09_30.xlsx')
GTdf = pd.read_excel(ExcelInput)
Y_pred = GTdf['Kangfu'].to_numpy()
Y_true = GTdf['Label'].to_numpy()
print('Accuracy:{}'.format(accuracy_score(y_true=Y_true, y_pred=Y_pred)))
target_names = ['N', 'Y']
print(classification_report(Y_true, Y_pred, target_names=target_names))