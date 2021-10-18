import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import cv2

from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import HeUniform

# Pretrain
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


class model_VGG16:
    def __init__(self, input_shape, epoch, batch_size):
        self.input_shape = input_shape
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate=1e-5)
        self.metrics = AUC()
        self.NAME = 'VGG-16-{}'.format(int(time.time()))
        self.callbacks = [ModelCheckpoint(filepath='model/best.h5', save_best_only=True, save_weights_only=True),
                          TensorBoard(log_dir='logs/{}'.format(self.NAME), histogram_freq=1),
                          EarlyStopping(),
                          ReduceLROnPlateau()]

    def build_model(self):
        pretrain = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        x = pretrain.output
        x = Flatten()(x)
        prediction = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=pretrain.input, outputs=prediction)

        self.model.compile(optimizer=self.optimizer, loss=self.focal_loss(), metrics=self.metrics)

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

    def train(self, train_generator, valid_generator):
        self.model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=self.epoch,
                       validation_data=valid_generator,
                       validation_steps=50,
                       callbacks=self.callbacks)

    def predict(self, list_test_file):
        result = []
        for i in range(len(list_test_file)):
            img = cv2.imread(list_test_file[i])
            img = cv2.resize(img, self.input_shape)
            result.append(self.model.predict(np.array([img])))
        return result
