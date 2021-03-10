from config import config
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.layers import LSTM,Concatenate,Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Reshape, ReLU, LeakyReLU, BatchNormalization, Flatten
from keras import backend as K
K.image_data_format() == "channels_last"

latent_dim = config['latent_dim']
num_class = config['num_class']

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

#%%
def get_lstm(timesteps, ROISignal):
    inputs = Input((ROISignal, timesteps))
    LSTMOutputList = list()
    for i in range(ROISignal):
        brain_region_signal = crop(dimension=1, start=i, end=i+1)(inputs)
        LSTMOutputList.append(LSTM(latent_dim)(brain_region_signal))
    ConResult = Concatenate(axis=-1)(LSTMOutputList)
    dens1 = Dense(100, activation='linear', kernel_initializer="he_normal")(ConResult)
    dens1 = Dropout(0.3)(dens1)
    dens2 = Dense(50, activation='linear', kernel_initializer="he_normal")(dens1)
    dens2 = Dropout(0.3)(dens2)
    dens3 = Dense(num_class, activation='sigmoid', kernel_initializer="he_normal")(dens2)

    model = Model(inputs=[inputs], outputs=[dens3])
    print(model.summary())
    return model

#%%
def get_conv1D(timesteps, ROISignal):

    inputs = Input((timesteps, ROISignal))

    x = Conv1D(100, 10, padding='same', activation='relu')(inputs)
    x = Conv1D(100, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(200, 10, padding='same', activation='relu')(x)
    x = Conv1D(200, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(3)(x)

    x = Conv1D(400, 10, padding='same', activation='relu')(x)
    x = Conv1D(400, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(2)(x)

    x = Conv1D(800, 5, padding='same', activation='relu')(x)
    x = Conv1D(800, 5, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(3)(x)

    x = Conv1D(800, 3, padding='same', activation='relu')(x)
    x = Conv1D(800, 3, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = AveragePooling1D(2)(x)        

    x = Flatten()(x)
    x = Dropout(0.5)(x)  

    dens1 = Dense(1000, activation='linear', kernel_initializer="he_normal")(x)
    dens1 = Dropout(0.3)(dens1)
    dens1 = Dense(500, activation='linear', kernel_initializer="he_normal")(dens1)
    dens1 = Dropout(0.3)(dens1)
    dens1 = Dense(num_class, activation='sigmoid', kernel_initializer="he_normal")(dens1)

    model = Model(inputs=[inputs], outputs=[dens1])
    # print(model.summary())
    return model

#%%
def get_conv1D_aux(timesteps, ROISignal, aux_len):

    main_input = Input((timesteps, ROISignal))
    aux_input = Input((aux_len,))

    x = Conv1D(100, 10, padding='same', activation='relu')(main_input)
    x = Conv1D(100, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(200, 10, padding='same', activation='relu')(x)
    x = Conv1D(200, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(3)(x)

    x = Conv1D(400, 10, padding='same', activation='relu')(x)
    x = Conv1D(400, 10, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(2)(x)

    x = Conv1D(800, 5, padding='same', activation='relu')(x)
    x = Conv1D(800, 5, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    # x = MaxPooling1D(3)(x)

    x = Conv1D(800, 3, padding='same', activation='relu')(x)
    x = Conv1D(800, 3, padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = AveragePooling1D(2)(x)        

    x = Flatten()(x)
    x = Dropout(0.5)(x)  

    dens1 = Dense(1000, activation='linear', kernel_initializer="he_normal")(x)
    dens1 = Dropout(0.3)(dens1)
    dens1 = Concatenate(axis=-1)([dens1, aux_input])
    dens1 = Dense(500, activation='linear', kernel_initializer="he_normal")(dens1)
    dens1 = Dropout(0.3)(dens1)
#    print('Dens1:{}'.format(K.int_shape(dens1)))
#    print('aux_input:{}'.format(K.int_shape(aux_input)))    
    dens1 = Dense(num_class, activation='sigmoid', kernel_initializer="he_normal")(dens1)

    model = Model(inputs=[main_input,aux_input], outputs=[dens1])
    # print(model.summary())
    return model

#%%
# def get_conv1D(timesteps, ROISignal):

#     inputs = Input((timesteps, ROISignal))

#     x = Conv1D(100, 10, padding='same')(inputs)
#     x = ReLU(x)
#     x = Conv1D(100, 10, padding='same')(x)
#     x = ReLU(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = MaxPooling1D(3)(x)

#     x = Conv1D(200, 10, padding='same')(x)
#     x = ReLU(x)
#     x = Conv1D(200, 10, padding='same')(x)
#     x = ReLU(x)
#     x = BatchNormalization(axis=-1)(x)
#     # x = MaxPooling1D(3)(x)

#     x = Conv1D(400, 10, padding='same')(x)
#     x = ReLU(x)
#     x = Conv1D(400, 10, padding='same')(x)
#     x = ReLU(x)
#     x = BatchNormalization(axis=-1)(x)
#     x = MaxPooling1D(3)(x)

#     x = Conv1D(800, 5, padding='same')(x)
#     x = ReLU(x)
#     x = Conv1D(800, 5, padding='same')(x)
#     x = ReLU(x)
#     x = BatchNormalization(axis=-1)(x)
#     # x = MaxPooling1D(3)(x)

#     x = Conv1D(800, 3, padding='same')(x)
#     x = ReLU(x)
#     x = Conv1D(800, 3, padding='same')(x)
#     x = ReLU(x)
#     x = BatchNormalization(axis=-1)(x)
#     # x = MaxPooling1D(3)(x)

#     x = Dropout(0.5)(x)
#     x = Reshape((-1,))(x)    

#     dens1 = Dense(1000, activation='linear', kernel_initializer="he_normal")(x)
#     dens1 = Dropout(0.3)(dens1)
#     dens1 = Dense(100, activation='linear', kernel_initializer="he_normal")(dens1)
#     dens1 = Dropout(0.3)(dens1)
#     dens1 = Dense(num_class, activation='sigmoid', kernel_initializer="he_normal")(dens1)

#     model = Model(inputs=[inputs], outputs=[dens1])
#     print(model.summary())
#     return model