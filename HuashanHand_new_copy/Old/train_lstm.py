from config import config
import os
import argparse
import numpy as np
from model import get_lstm
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
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()

IFUseWeight = False
timesteps = config['timesteps']
ROINum = config['ROINum']

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
step = args.currFold

# train
def train_and_predict(datastore,tempStore):
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30) 
        
    X_train = np.load(os.path.join(datastore, 'x_train_{}.npy'.format(step)))
    X_train = np.transpose(X_train,(0,2,1))
    y_train = np.load(os.path.join(datastore, 'y_train_{}.npy'.format(step)))
    X_train = X_train.astype('float32')
    print('x_train: shape{}'.format(X_train.shape))
    print('y_train: shape{}'.format(y_train.shape))
    # convert class vectors to binary class matrices
    nb_classes = len(np.unique(y_train))
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #---------------------------------#
    model = get_lstm(timesteps, ROINum)
    #---------------------------------#
    weightDir = os.path.join(tempStore, 'weights.h5')
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    if os.path.exists(weightDir) and IFUseWeight == True:
        model.load_weights(os.path.join(tempStore, 'weights.h5'))
    train_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,\
     verbose=1, shuffle=True,validation_split=0.2, callbacks=[model_checkpoint, early_stop])
    
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    np.save(os.path.join(tempStore,'loss.npy'),loss)
    np.save(os.path.join(tempStore,'val_loss.npy'),val_loss)
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    X_test = np.load(os.path.join(datastore, 'x_test_{}.npy'.format(step)))
    X_test = np.transpose(X_test,(0,2,1))
    X_test = X_test.astype('float32')
    y_test = np.load(os.path.join(datastore, 'y_test_{}.npy'.format(step)))
    # convert class vectors to binary class matrices
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model = get_lstm(timesteps, ROINum)
    model.load_weights(os.path.join(tempStore, 'weights.h5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    np.save(os.path.join(tempStore,'Y_predict.npy'), Y_predict) 

def value_predict(X_test, load_weight_dir, outputDir=None):   
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)      
    X_test = X_test.astype('float32')
    
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model = get_lstm(timesteps, ROINum)
    model.load_weights(load_weight_dir)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    Y_predict = model.predict(X_test, verbose=1)
    if outputDir != None:
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)
            np.save(os.path.join(outputDir,'Y_predict.npy'), Y_predict)
    return Y_predict 

if __name__ == '__main__':
    datastore = './dataStore'
    tempStore = './tempData'
    if not os.path.exists(tempStore):
        os.mkdir(tempStore)
    train_and_predict(datastore,tempStore)