from DataGenerator import customDataGenerator
from load_data import DataLoader
from model import model_VGG16
import numpy as np
import time
from CBAM_ResNet import CBAM_Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

if __name__ == "__main__":
    dataloader = DataLoader(df_train_path='dataframe/train.csv', df_test_path='dataframe/sample_submission.csv',
                            file_train='data/train/', file_test='data/test/')

    x_train, x_valid, y_train, y_valid = dataloader.train_loader(major_rate=0.5)
    x_test = dataloader.test_loader()

    train_gen = customDataGenerator(batch_size=32, list_file_name=x_train, labels=y_train, n_classes=2, dim=224,
                                    shuf=True, scale=True, rate=1, is_valid=False, aug=False)

    val_gen = customDataGenerator(batch_size=32, list_file_name=x_valid, labels=y_valid, n_classes=2, dim=224,
                                  is_valid=True, aug=False)

    optimizer = Adam(learning_rate=1e-5)
    metrics = AUC()
    NAME = 'VGG-16-{}'.format(int(time.time()))
    callbacks = [ModelCheckpoint(filepath='model/best.h5', save_best_only=True, save_weights_only=True),
                      TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=1),
                      EarlyStopping(),
                      ReduceLROnPlateau()]
    model = CBAM_Model(input_shape=(224,224,3), epoch=2, batch_size=32, optimizer=optimizer, metric=metrics, callbacks=callbacks)
    model.build()
    #vgg16.train(train_generator=train_gen, valid_generator=val_gen)
