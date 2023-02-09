import keras
from keras.layers import Dense, Convolution1D, Convolution2D, MaxPool1D, MaxPool2D, Flatten, Dropout
from keras import optimizers, losses, activations, models
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D,GlobalMaxPool1D, Softmax, Add, Flatten, Activation, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics


def network():
    inp = Input(shape = (127, 1))
    
    conv1_1 = Convolution1D(16, (5))(inp)
    conv1_1 = BatchNormalization()(conv1_1)
    A1_1 = Activation("relu")(conv1_1)
    conv1_2 = Convolution1D(16, (5))(A1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    A1_2 = Activation("relu")(conv1_2)
    pool1 = MaxPool1D(pool_size = (2), strides = (2), padding = 'same')(A1_2)
    
    conv2_1 = Convolution1D(32, (3))(pool1)
    conv2_1 = BatchNormalization()(conv2_1)
    A2_1 = Activation("relu")(conv2_1)
    conv2_2 = Convolution1D(32, (3))(A2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    A2_2 = Activation("relu")(conv2_2)
    pool2 = MaxPool1D(pool_size = (2), strides = (2), padding = 'same')(A2_2)
    
    conv3_1 = Convolution1D(64, (3))(pool2)
    conv3_1 = BatchNormalization()(conv3_1)
    A3_1 = Activation("relu")(conv3_1)
    conv3_2 = Convolution1D(64, (3))(A3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    A3_2 = Activation("relu")(conv3_2)
    pool3 = MaxPool1D(pool_size = (2), strides = (3), padding = 'same')(A3_2)
    
    flatten = Flatten()(pool3)
    
    dense_end1 = Dense(64, activation = 'relu')(flatten)
    A_end = BatchNormalization()(dense_end1)
    dense_end2 = Dense(32, activation = 'relu')(A_end)
    main_output = Dense(4, activation = 'softmax')(dense_end2)
    
    model = Model(inputs= inp, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
    
    model.summary()
    
    return model