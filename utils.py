from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, Bidirectional, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Concatenate, MaxPooling1D ,ConvLSTM2D
from tensorflow.keras.layers import Lambda, Reshape, Activation, Add, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling1D
# from tensorflow.keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from itertools import cycle
import keras
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.figure(figsize=(15,15))
        plt.subplot(211)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.subplot(212)
        plt.plot(self.x, self.accs, label="acc")
        plt.plot(self.x, self.val_accs, label="val_acc")
        plt.legend()
        plt.show();

def pool_1D(inputs, filters, kernel_size=3):
    conv = Conv1D(filters, (kernel_size), padding='same')(inputs)
    conv = Activation('relu')(conv)
    conv = Dropout(0.2)(conv)
    conv = AveragePooling1D()(conv)
    return conv

def upsample_1D(inputs, residual, filters, kernel_size=3):
    conv = Conv1D(filters, (kernel_size), padding='same')(inputs)
    residual = Conv1D(filters, (1))(residual)
    conv = Add()([conv,residual])
    conv = Activation('relu')(conv)
    conv = UpSampling1D()(conv)
    return conv

def build_hourglass_model(input_shape, conv_filter_size=3, conv_kernel_size=3, n_pooling_upsampling_steps=1, n_stacks=1, n_classes=4):
    input_layer = Input(shape=input_shape, name='input')
    x = pool_1D(input_layer, conv_filter_size)
    res1 = x
    outputs = []
    
    for n in range(n_stacks):
        for idx in range(n_pooling_upsampling_steps):
            x = pool_1D(x, conv_filter_size*(idx+1), kernel_size=conv_kernel_size)
            outputs.append(x)

        for idx in range(n_pooling_upsampling_steps):
            x = upsample_1D(x, outputs[-(idx+1)], conv_filter_size*(n_pooling_upsampling_steps-idx), kernel_size=conv_kernel_size)

    x = upsample_1D(x, res1,  conv_filter_size)
    x = Conv1D(n_classes, (conv_kernel_size), padding='same')(x)
    x = Activation('relu')(x)
    model = Model(input_layer, x)
    return model

def build_basic_lstm_model(input_shape=None, nb_classes=4, lstm_filters=32, dropout=0.5, use_bidirectional=False):    
    inputs = Input(shape=input_shape, name='input')
    if use_bidirectional is True:
        x = Bidirectional(LSTM(lstm_filters, return_sequences=True))(inputs)
    else:
        x = LSTM(lstm_filters, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(nb_classes, activation='sigmoid'))(x)
    model = Model(inputs, x)
    return model

def build_conv_lstm_model(input_shape=None, nb_classes=4, lstm_filters=100, conv_filters=64, conv_kernel_size=1, dropout=0.5, pool_size=2):    
    inputs = Input(shape=input_shape, name='input')
    x = TimeDistributed(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))(inputs)
    x = TimeDistributed(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))(x)
    x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=pool_size))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(lstm_filters)(x)
    x = Dropout(dropout)(x)
    x = Dense(lstm_filters, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs, x)
    return model

def build_conv_lstm_2d_model(input_shape=None, nb_classes=4, conv_filters=8, conv_kernel_size=(1,1), lstm_filters=20, dropout=0.5):
    inputs = Input(shape=input_shape, name='input')
    x = ConvLSTM2D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', recurrent_dropout=dropout)(inputs)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(lstm_filters, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs, x)
    return model


def build_perceptual_model():
    weights = None
    if weights is None:
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )
        feature_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )
    else:
        feature_model = load_model(weights)
        feature_model.layers.pop()
        feature_model.layers.pop()  # get to pool layer
        feature_model.outputs = [self.model.layers[-1].output]
        feature_model.output_layers = [self.model.layers[-1]]
        feature_model.layers[-1].outbound_nodes = []
    return feature_model


def extract(model, image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the prediction.
    features = model.predict(x)

    features = features[0]

    return features


def evenly_subsample_features(labels, features, idx, counts, seq_len, seq_stps, n_features):
    c = cycle([i for i in np.unique(idx)])
    n_examples = int(np.min(counts)-seq_len)
    if seq_steps == None:
        X = np.empty((n_examples*len(np.unique(idx)), n_features))
        Y = np.empty((n_examples*len(np.unique(idx)), labels.shape[1]))
        for j in range(n_examples*len(np.unique(idx))):
            i = next(c)
            try:
                ix = np.random.choice(np.where(idx==i)[0])
                X[j, :] = features[ix]
                Y[j, :] = labels[ix]
            except ValueError:
                ix = np.random.choice(np.where(idx==i)[0])
                X[j, :] = features[ix]
                Y[j, :] = labels[ix]
    else:
        if seq_stps == 0:
            X = np.empty((n_examples*len(np.unique(idx)), seq_len, n_features))
            Y = np.empty((n_examples*len(np.unique(idx)), seq_len, labels.shape[1]))
            for j in range(n_examples*len(np.unique(idx))):
                i = next(c)
                try:
                    ix = np.random.choice(np.where(idx==i)[0])
                    X[j, :, :] = features[ix-int(seq_len/2):
                                        ix+int(seq_len/2)]
                    Y[j, :, :] = labels[ix-int(seq_len/2):
                                        ix+int(seq_len/2)]
                except ValueError:
                    ix = np.random.choice(np.where(idx==i)[0])
                    X[j, :, :] = features[ix-int(seq_len/2):
                                        ix+int(seq_len/2)]
                    Y[j, :, :] = labels[ix-int(seq_len/2):
                                        ix+int(seq_len/2)]
        else:
            X = np.empty((n_examples*len(np.unique(idx)),
                        seq_stps,
                        int(seq_len/seq_stps),
                        n_features))
            
            Y = np.empty((n_examples*len(np.unique(idx)), labels.shape[1]))
            for j in range(n_examples*len(np.unique(idx))):
                i = next(c)
                try:
                    ix = np.random.choice(np.where(idx==i)[0])
                    X[j, :, :] = features[ix-int(seq_len/2):
                                        ix+int(seq_len/2)].reshape((seq_stps,
                                                int(seq_len/seq_stps),
                                                n_features))
                    Y[j, :] = labels[ix]
                except ValueError:
                    ix = np.random.choice(np.where(idx==i)[0])
                    X[j, :, :] = features[ix-int(seq_len/2):
                                        ix+int(seq_len/2)].reshape((seq_stps,
                                                int(seq_len/seq_stps),
                                                n_features))
                    Y[j, :] = labels[ix]
    return X, Y
