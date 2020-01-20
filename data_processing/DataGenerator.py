import numpy as np
import cv2 as cv2
import os
import random
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, image_path, mask_path,
                 to_fit=True, batch_size=8, dim=(256, 256),
                 n_channels=1, n_classes=1, shuffle=True):
        """Initialization
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.img_labels = os.listdir(self.image_path)
        self.mask_labels = os.listdir(self.mask_path)
        self.img_labels.sort(key=lambda f: [int(s) for s in f[:-4].split('_') if s.isdigit()][0])
        self.mask_labels.sort(key=lambda f: [int(s) for s in f[:-4].split('_') if s.isdigit()][0])
        self.list_IDs = list(range(len(self.img_labels)))
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self._generate_X(list_IDs_temp)
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        grayscale = self.n_channels == 1
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = self._load_image(self.image_path + self.img_labels[ID], grayscale=grayscale)
            X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.n_channels))

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        grayscale = self.n_classes==1
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # y[i,] = self._load_grayscale_image(self.mask_path + self.mask_labels[ID])
            y = self._load_image(self.mask_path + self.mask_labels[ID], grayscale=grayscale)
            Y[i,] = np.reshape(y, (self.dim[0], self.dim[1], self.n_classes))
        return Y

    def _load_image(self, image_path, grayscale=False):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=self.dim, interpolation=cv2.INTER_CUBIC)
        img = img / 255
        return img
