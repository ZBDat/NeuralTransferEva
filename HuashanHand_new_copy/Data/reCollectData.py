#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:17:58 2019

@author: zhaoyu
"""

import os
import shutil
import glob

originalPath = '/media/data/yuzhao/data/HuashanHand'
newPath = '/media/data/yuzhao/data/HuaShanHandNew'
FunCheckItem = ['BDREST','BOLD']
T1CheckItem = ['3DT1']

FunRaw = os.path.join(newPath,'FunRaw')
T1Raw = os.path.join(newPath, 'T1Raw')

if not os.path.exists(newPath):
    os.mkdir(newPath)
if not os.path.exists(FunRaw):
    os.mkdir(FunRaw)
if not os.path.exists(T1Raw):
    os.mkdir(T1Raw)

originFileList = os.listdir(originalPath)
selectedFileList = []
for item in originFileList:
    if item=='.' or item=='..' or item == '.DS_Store' or item =='分组列表编号信息.xlsx':
        pass
    elif item.isdigit():
        selectedFileList.append(item)
    else:
        raise ValueError('Please check' + item)

for ID in selectedFileList:
    patientID = int(ID)
    SubFunRaw = os.path.join(FunRaw,'Sub_' + '%03d' % patientID)
    SubT1Raw = os.path.join(T1Raw,'Sub_' + '%03d' % patientID)
    if not os.path.exists(SubFunRaw):
        os.mkdir(SubFunRaw)
    if not os.path.exists(SubT1Raw):
        os.mkdir(SubT1Raw)
    allFilesInPatient = glob.glob(os.path.join(originalPath,ID)+'/*/*/*.dcm')
    for item in allFilesInPatient:
        Bname = os.path.basename(item)
        IFFUNLIST = [checkitem in item for checkitem in FunCheckItem]
        IFT1LIST = [checkitem in item for checkitem in T1CheckItem]
        if True in IFT1LIST:
            shutil.copyfile(item, os.path.join(SubT1Raw,Bname))        
        elif True in IFFUNLIST:
            shutil.copyfile(item, os.path.join(SubFunRaw,Bname))
    del allFilesInPatient
    print('Current patient is : ' + str(patientID))

    

    