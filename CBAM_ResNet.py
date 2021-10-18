import numpy as np
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Flatten, MaxPool2D, AvgPool2D, \
    Add, Lambda, Concatenate, multiply, Multiply, ZeroPadding2D, Activation
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K


# Pretrain


class CBAM_Model:
    def __init__(self, input_shape, epoch, batch_size, optimizer, callbacks, metric):
        self.input_shape = input_shape
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.metric = metric

    def CBAM(self, x, r):
        '''MLP block'''
        mlp = Sequential([
            Dense(x.shape[3] // r, activation='relu'),
            Dense(x.shape[3])
        ])

        '''Channel attention block'''
        x_max_pool = MaxPool2D(pool_size=(x.shape[1], x.shape[1]))(x)
        x_max_pool = mlp(x_max_pool)

        x_avg_pool = AvgPool2D(pool_size=(x.shape[1], x.shape[1]))(x)
        x_avg_pool = mlp(x_avg_pool)

        channel = sigmoid(Add()([x_avg_pool, x_max_pool]))

        x = multiply([x, channel])

        '''Spatial attention block'''
        mean_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
        x_concat = Concatenate(axis=3)([mean_pool, max_pool])
        spatial = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(x_concat)
        return Multiply()([x, spatial])
    def identity_block(self, X, kernel_size, filter):
        F1, F2, F3 = filter

        X_shortcut = X

        X = Conv2D(F1, kernel_size=1, strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(F2, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(F3, kernel_size=1, strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)

        X = Add()([X, X_shortcut])
        return Activation('relu')(X)

    def conv_block(self, X, kernel_size, filter, stride):
        F1, F2, F3 = filter

        X_shortcut = X

        X = Conv2D(F1, kernel_size=1, strides=stride,  kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=kernel_size, strides=1, padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization()(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=stride, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)

        X = Add()([X, X_shortcut])
        return Activation('relu')(X)
    def focal_loss(self, alpha=0.25, gamma=2.0):
        def focal_crossentropy(y_true, y_pred):
            y_pred = tf.cast(y_pred, tf.float32)
            y_true = tf.cast(y_true, tf.float32)
            bce = K.binary_crossentropy(y_true, y_pred)

            y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
            p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

            alpha_factor = 1
            modulating_factor = 1

            alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
            modulating_factor = K.pow((1 - p_t), gamma)

            # compute the final loss and return
            return K.mean(alpha_factor * modulating_factor * bce, axis=-1)

        return focal_crossentropy

    def build(self):
        input = Input(shape=self.input_shape)
        #X = ZeroPadding2D(3, 3)(input)

        #Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPool2D((3, 3), strides=(2, 2))(X)
        X = self.CBAM(X, 4)
        #Stage 2
        X = self.conv_block(X, kernel_size=3, filter=[64, 64, 256],stride=1)
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.CBAM(X, 4)
        #Stage 3
        X = self.conv_block(X, kernel_size=3, filter=[128, 128, 512], stride=2)
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.CBAM(X, 4)
        #Stage 4
        X = self.conv_block(X, kernel_size=3, filter=[256, 256, 1024], stride=2)
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.CBAM(X, 4)
        #Stage 5
        X = self.conv_block(X, kernel_size=3, filter=[512, 512, 2048], stride=2)
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = self.identity_block(X, 3, [512, 512, 2048])

        X = AvgPool2D((2, 2))(X)
        X = Flatten()(X)
        output = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0))(X)

        self.model = Model(inputs=input, outputs = output)
        self.model.compile(optimizer=self.optimizer, loss=self.focal_loss(), metrics=self.metric)
        self.model.summary()
        return self.model


    def train(self, train_gen, valid_gen):
        self.model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen), validation_steps=len(valid_gen), epochs = self.epoch,
                       callbacks=self.callbacks)

    def predict(self, list_test_file):
        result = []
        for i in range(len(list_test_file)):
            img = cv2.imread(list_test_file[i])
            img = cv2.resize(img, self.input_shape)
            result.append(self.model.predict(np.array([img])))
        return result
