from __future__ import print_function

from config import config
import os
import argparse
import numpy as np
from model import get_lstm, get_conv1D, get_conv1D_aux
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras import backend as K

K.image_data_format() == "channels_last"

parser = argparse.ArgumentParser(description='PyTorch LNM bags Pipeline')
parser.add_argument('--foldNum', type=int, default=6, metavar='fN',
                    help='number of folds for cross validation (default: 6)')
parser.add_argument('--currFold', type=int, default=1, metavar='cF',
                    help='current fold of the cross validation (default: 1)')
parser.add_argument('--batch-size', type=int, default=6, metavar='B',
                    help='batch size (default: 16)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--use-aux', action='store_true', default=False,
                    help='Do not use auxInfo')

args = parser.parse_args()

IFUseWeight = False
timesteps = config['timesteps']
ROINum = config['ROINum']
aux_len = config['aux_len']

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
Is_use_aux = args.use_aux
step = args.currFold


# train
def train_and_predict(datastore, tempStore):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    X_train = np.load(os.path.join(datastore, 'x_train_{}.npy'.format(step)))
    X_train = X_train.astype('float32')
    print('x_train: shape{}'.format(X_train.shape))

    aux_train = np.load(os.path.join(datastore, 'aux_train_{}.npy'.format(step)))
    aux_train = aux_train.astype('float32')
    print('aux_train: shape{}'.format(aux_train.shape))

    # convert class vectors to binary class matrices
    y_train = np.load(os.path.join(datastore, 'y_train_{}.npy'.format(step)))
    print('y_train: shape{}'.format(y_train.shape))
    nb_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    X_test = np.load(os.path.join(datastore, 'x_test_{}.npy'.format(step)))
    X_test = X_test.astype('float32')
    aux_test = np.load(os.path.join(datastore, 'aux_test_{}.npy'.format(step)))
    aux_test = aux_test.astype('float32')
    y_test = np.load(os.path.join(datastore, 'y_test_{}.npy'.format(step)))
    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    # ---------------------------------#
    if Is_use_aux:
        model = get_conv1D_aux(timesteps, ROINum, aux_len)
        X_train = [X_train, aux_train]
        X_test = [X_test, aux_test]
    else:
        model = get_conv1D(timesteps, ROINum)
        X_train = X_train
        X_test = X_test
    # ---------------------------------#
    weightDir = os.path.join(tempStore, 'weights_{}.h5'.format(step))
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint(weightDir, monitor='val_acc', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=200, verbose=0, mode='max')
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    if os.path.exists(weightDir) and IFUseWeight == True:
        model.load_weights(os.path.join(tempStore, 'weights_{}.h5'.format(step)))
    train_history = model.fit(X_train, Y_train, batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              shuffle=True,
                              validation_data=(X_test, Y_test),
                              callbacks=[model_checkpoint, early_stop])

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    np.save(os.path.join(tempStore, 'loss_{}.npy'.format(step)), loss)
    np.save(os.path.join(tempStore, 'val_loss_{}.npy'.format(step)), val_loss)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    score = model.evaluate(X_test, Y_test, verbose=0)
    model.save_weights(os.path.join(tempStore, 'last_weights_{}.h5'.format(step)))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    # ---------------------------------#
    if Is_use_aux:
        model = get_conv1D_aux(timesteps, ROINum, aux_len)
    else:
        model = get_conv1D(timesteps, ROINum)
    # ---------------------------------#
    model.load_weights(os.path.join(tempStore, 'weights_{}.h5'.format(step)))

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    Y_predict = model.predict(X_test, verbose=1)
    np.save(os.path.join(tempStore, 'Y_predict_{}.npy'.format(step)), Y_predict)


def value_predict(X_test, aux_test, load_weight_dir, outputDir=None, Is_use_aux=False):
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    X_test = X_test.astype('float32')
    aux_test = aux_test.astype('float32')

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    # ---------------------------------#
    if Is_use_aux:
        model = get_conv1D_aux(timesteps, ROINum, aux_len)
        X_test = [X_test, aux_test]
    else:
        model = get_conv1D(timesteps, ROINum)
        X_test = X_test
    # ---------------------------------#
    model.load_weights(load_weight_dir)

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    Y_predict = model.predict(X_test, verbose=1)
    if outputDir != None:
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            np.save(os.path.join(outputDir, 'Y_predict_{}.npy'.format(step)), Y_predict)
    return Y_predict


if __name__ == '__main__':
    datastore = './dataStore'
    if Is_use_aux:
        tempStore = './tempData/withAux'
    else:
        tempStore = './tempData/withoutAux'
    if not os.path.exists(tempStore):
        os.makedirs(tempStore)
    train_and_predict(datastore, tempStore)
