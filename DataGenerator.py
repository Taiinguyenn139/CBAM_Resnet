import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
import cv2



class customDataGenerator(Sequence):
    def __init__(self, batch_size, list_file_name, labels, n_classes, dim, shuf=False, aug=False, scale=True,
                 rate=1, is_valid=False):
        self.batch_size = batch_size
        self.list_file_name = list_file_name
        self.labels = labels
        self.shuf = shuf
        self.aug = aug
        self.n_classes = n_classes
        self.dim = dim
        self.scale = scale
        self.rate = rate
        self.is_valid = is_valid
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.list_file_name) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.list_file_name[item * self.batch_size: (item + 1) * self.batch_size]
        batch_y = self.labels[item * self.batch_size: (item + 1) * self.batch_size]

        x, y = self.data_generator(batch_x, batch_y)

        if self.is_valid:
            return np.array(x, dtype=float), y
        if self.aug:
            x = self.augment(x)

        return np.array(x, dtype=float), y

    def on_epoch_end(self):
        if self.shuf:
            self.list_file_name, self.labels = shuffle(self.list_file_name, self.labels)

    def data_generator(self, batch_x, batch_y):
        x = np.empty(shape=(self.batch_size, self.dim, self.dim, 3))
        y = np.empty(shape=(self.batch_size), dtype=int)

        for i in range(self.batch_size):
            img = cv2.imread(self.list_file_name[i])
            img = cv2.resize(img, (self.dim, self.dim))
            x[i] = img
            y[i] = self.labels[i]

        if self.scale:
            return np.array(x, dtype=float)/255., np.array(y)
        return np.array(x, dtype=float), np.array(y)

    def augment(self, x):
        sometimes = lambda aug : iaa.Sometimes(self.rate, aug) # sometimes (rate, ...) applies the given augmenter in rate of all case
        seq = iaa.Sequential(
            [
                iaa.Fliplr(self.rate),
                sometimes(iaa.Affine(
                    rotate=(-20, 20),
                    translate_percent={"x":(-0.2, 0.2), "y":(-0.2, 0.2)}
                )),
            ], random_order=True
        )
        return seq.augment_image(x)