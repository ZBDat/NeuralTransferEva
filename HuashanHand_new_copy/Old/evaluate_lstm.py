from __future__ import print_function

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from train import value_predict

tempStore = './tempData'
datastore = './dataStore'
step = '1'

#%%
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC
#%%
if __name__ == "__main__":

    print('-'*60)
    print('test result')
    # predict result
    y_possibility = np.load(os.path.join(tempStore,'Y_predict.npy'))
    print ('Y_possibility:{}'.format(y_possibility))
    Y_predict = np.argmax(y_possibility, axis=1)
    print ('Y_predict:{}'.format(Y_predict))
    print ('Y_predict_type:{}'.format(Y_predict.dtype))

    # ground truth
    Y_test = np.load(os.path.join(datastore, 'y_test_' + step + '.npy'))
    Y_test = np.squeeze(Y_test)
    Y_test= np.int64(Y_test)
    print ('Y_test_:{}'.format(Y_test))
    print ('Y_test_type:{}'.format(Y_test.dtype))

    #classification_report
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_test, Y_prob = y_possibility[:,1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_test, Y_predict, target_names=target_names))

    #%%

    print('-'*60)
    print('training result')
    
    # predict result
    X_train = np.load(os.path.join(datastore, 'x_train_' + step + '.npy'))
    load_weight_dir = os.path.join(tempStore, 'weights.h5')
    X_train = np.transpose(X_train,(0,2,1))
    y_possibility = value_predict(X_train, load_weight_dir, outputDir=None)
    print ('Y_possibility:{}'.format(y_possibility))
    Y_predict = np.argmax(y_possibility, axis=1)
    print ('Y_predi:{}'.format(Y_predict))
    print ('Y_predict_type:{}'.format(Y_predict.dtype))

    # ground truth
    y_train = np.load(os.path.join(datastore, 'y_train_' + step + '.npy'))
    Y_train = np.squeeze(y_train)
    Y_train= np.int64(Y_train)
    print ('Y_train:{}'.format(Y_train))
    print ('Y_train_type:{}'.format(Y_train.dtype))

    #classification_report
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_train, Y_prob = y_possibility[:,1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_train, Y_predict, target_names=target_names))