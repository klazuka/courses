import numpy as np
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args:
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class FineTunedVgg16():
    """
        The VGG 16 Imagenet model
    """

    def __init__(self, num_classes):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create(num_classes)

    def ConvBlock(self, layers, filters):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be
                                created for each layer.
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def create(self, num_classes):
        """
            Creates the VGG16 network architecture and loads the pretrained weights.

            Args: num_classes the number of output classes
            Returns:   None
        """
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH + fname, cache_subdir='models'))

        self._finetune(num_classes)

    def _finetune(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        model = self.model

        model.pop()
        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(num, activation='softmax'))

    def compile(self, optimizer):
        """
            Compile the model using categorical cross-entropy loss

        :param optimizer: how to optimize the loss function
        :return: None
        """
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, train_batches, validation_batches, num_epochs=1, callbacks=None):
        """
        Train the model on `train_batches` using `validation_batches` for validation

        :param train_batches: a generator that yields (images, labels) pairs
        :param validation_batches: same as `train_batches`
        :param num_epochs: number of epochs to train on
        :param callbacks: list of keras.callbacks.Callback objects
        :return: Keras History object
        """
        return self.model.fit_generator(train_batches,
                                        samples_per_epoch=train_batches.nb_sample,
                                        nb_epoch=num_epochs,
                                        validation_data=validation_batches,
                                        nb_val_samples=validation_batches.nb_sample,
                                        callbacks=callbacks)