from __future__ import print_function

from config import config
import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch LNM bags Pipeline')
parser.add_argument('--foldNum', type=int, default=6, metavar='fN',
                    help='number of folds for cross validation (default: 6)')
parser.add_argument('--currFold', type=int, default=1, metavar='cF',
                    help='current fold of the cross validation (default: 1)')
args = parser.parse_args()
step = args.currFold

#%%
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC
# train
def train_and_predict(datastore,tempStore):
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30) 
        
    aux_train = np.load(os.path.join(datastore, 'aux_train_{}.npy'.format(step))) 
    aux_train = aux_train.astype('float32')
    print('aux_train: shape{}'.format(aux_train.shape))    
    # convert class vectors to binary class matrices
    y_train = np.load(os.path.join(datastore, 'y_train_{}.npy'.format(step)))
    print('y_train: shape{}'.format(y_train.shape))
    # nb_classes = len(np.unique(y_train))
    # Y_train = np_utils.to_categorical(y_train, nb_classes)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    aux_test = np.load(os.path.join(datastore, 'aux_test_{}.npy'.format(step)))
    aux_test = aux_test.astype('float32')
    y_test = np.load(os.path.join(datastore, 'y_test_{}.npy'.format(step)))
    # convert class vectors to binary class matrices
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #---------------------------------#
    model = SVC(C=1, 
                kernel='linear', 
                degree=3, 
                gamma='auto', 
                coef0=0.0, 
                shrinking=True, 
                probability=True, 
                tol=0.001, 
                cache_size=200, 
                class_weight=None, 
                verbose=False, 
                max_iter=-1, 
                random_state=None
                )
    #---------------------------------#
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(aux_train, y_train)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    y_possibility = model.predict_proba(aux_test)
    np.save(os.path.join(tempStore,'Y_predict.npy'), y_possibility) 

    # evaluate
    print('-'*60)
    print('best test result')
    # predict result
    Y_predict = np.argmax(y_possibility, axis=1)    
    print ('Y_predict_type:{}'.format(Y_predict.dtype))
    print ('Y_predi:{}'.format(Y_predict))

    # ground truth
    Y_test = np.squeeze(y_test)
    Y_test= np.int64(Y_test)
    print ('Y_test_:{}'.format(Y_test))
    print ('Y_test_type:{}'.format(Y_test.dtype))

    #classification_report
    print('Accuracy:{}'.format(accuracy_score(y_true=Y_test, y_pred=Y_predict)))
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_test, Y_prob = y_possibility[:,1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_test, Y_predict, target_names=target_names))

if __name__ == '__main__':
    datastore = './dataStore'
    tempStore = './tempData'
    if not os.path.exists(tempStore):
        os.mkdir(tempStore)
    train_and_predict(datastore,tempStore)