from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score
from train_conv import value_predict

parser = argparse.ArgumentParser(description='PyTorch LNM bags Pipeline')
parser.add_argument('--foldNum', type=int, default=6, metavar='fN',
                    help='number of folds for cross validation (default: 6)')
parser.add_argument('--currFold', type=int, default=1, metavar='cF',
                    help='current fold of the cross validation (default: 1)')
parser.add_argument('--use-aux', action='store_true', default=False,
                    help='Do not use auxInfo')

args = parser.parse_args()

step = str(args.currFold)
Is_use_aux = args.use_aux

datastore = './dataStore'
if Is_use_aux:
    tempStore = './tempData/withAux'
else:
    tempStore = './tempData/withoutAux'


# %%
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC


# %%
if __name__ == "__main__":
    print('-' * 60)
    print('best test result')
    # predict result
    y_possibility = np.load(os.path.join(tempStore, 'Y_predict_{}.npy'.format(step)))
    # print ('Y_possibility:{}'.format(y_possibility))
    Y_predict = np.argmax(y_possibility, axis=1)
    print('Y_Poss:' + str(y_possibility.shape))
    print('Y_Pred:' + str(Y_predict.shape))
    print('Y_predict_type:{}'.format(Y_predict.dtype))
    print('Y_predi:{}'.format(Y_predict))

    df_1 = pd.read_csv(os.path.join(datastore, 'test_fold_{}.csv'.format(step)))
    output_file = os.path.join(tempStore, 'test_fold_{}.csv'.format(step))
    df_2 = pd.DataFrame({'Y_Pred': Y_predict, 'Y_Poss_0': y_possibility[:, 0], 'Y_Poss_1': y_possibility[:, 1]})
    df_new = pd.concat([df_1, df_2], axis=1)
    df_new.to_csv(output_file, index=None)

    # ground truth
    Y_test = np.load(os.path.join(datastore, 'y_test_' + step + '.npy'))
    Y_test = np.squeeze(Y_test)
    Y_test = np.int64(Y_test)
    print('Y_test_:{}'.format(Y_test))
    print('Y_test_type:{}'.format(Y_test.dtype))

    # classification_report
    print('Accuracy:{}'.format(accuracy_score(y_true=Y_test, y_pred=Y_predict)))
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_test, Y_prob=y_possibility[:, 1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_test, Y_predict, target_names=target_names))

    # %%
    print('-' * 60)
    print('Last test result')

    # predict result
    X_test = np.load(os.path.join(datastore, 'x_test_' + step + '.npy'))
    aux_test = np.load(os.path.join(datastore, 'aux_test_{}.npy'.format(step)))
    load_weight_dir = os.path.join(tempStore, 'last_weights_{}.h5'.format(step))

    y_possibility = value_predict(X_test, aux_test, load_weight_dir, outputDir=None, Is_use_aux=Is_use_aux)
    # print ('Y_possibility:{}'.format(y_possibility))
    Y_predict = np.argmax(y_possibility, axis=1)
    print('Y_predict_type:{}'.format(Y_predict.dtype))
    print('Y_predi:{}'.format(Y_predict))

    # ground truth
    y_test = np.load(os.path.join(datastore, 'y_test_' + step + '.npy'))
    Y_test = np.squeeze(y_test)
    Y_test = np.int64(Y_test)
    print('Y_test_:{}'.format(Y_test))
    print('Y_test_type:{}'.format(Y_test.dtype))

    # classification_report
    print('Accuracy:{}'.format(accuracy_score(y_true=Y_test, y_pred=Y_predict)))
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_test, Y_prob=y_possibility[:, 1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_test, Y_predict, target_names=target_names))

    # %%
    print('-' * 60)
    print('training result')

    # predict result
    X_train = np.load(os.path.join(datastore, 'x_train_' + step + '.npy'))
    aux_train = np.load(os.path.join(datastore, 'aux_train_{}.npy'.format(step)))
    load_weight_dir = os.path.join(tempStore, 'weights_{}.h5'.format(step))

    y_possibility = value_predict(X_train, aux_train, load_weight_dir, outputDir=None, Is_use_aux=Is_use_aux)
    # print ('Y_possibility:{}'.format(y_possibility))
    Y_predict = np.argmax(y_possibility, axis=1)
    print('Y_predict_type:{}'.format(Y_predict.dtype))
    print('Y_predi:{}'.format(Y_predict))

    df_1 = pd.read_csv(os.path.join(datastore, 'train_fold_{}.csv'.format(step)))
    output_file = os.path.join(tempStore, 'train_fold_{}.csv'.format(step))
    df_2 = pd.DataFrame({'Y_Pred': Y_predict, 'Y_Poss_0': y_possibility[:, 0], 'Y_Poss_1': y_possibility[:, 1]})
    df_new = pd.concat([df_1, df_2], axis=1)
    df_new.to_csv(output_file, index=None)

    # ground truth
    y_train = np.load(os.path.join(datastore, 'y_train_' + step + '.npy'))
    Y_train = np.squeeze(y_train)
    Y_train = np.int64(Y_train)
    print('Y_train:{}'.format(Y_train))
    print('Y_train_type:{}'.format(Y_train.dtype))

    # classification_report
    print('ROCAUC:{}'.format(calculate_ROCAUC(Y_ture=Y_train, Y_prob=y_possibility[:, 1])))
    target_names = ['Y', 'N']
    print(classification_report(Y_train, Y_predict, target_names=target_names))
