from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from agnez.keras_callbacks import Grid2D, Plot

'''
    Borrowed from keras/examples/mlp_mnist.py

    Train a simple deep NN on the MNIST dataset.
    Get to 98.30% test accuracy after 20 epochs (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a GRID K520 GPU.

    In order to use Bokeh plotting here, you have to have a bokeh-server loaded
    up. For example, run on the terimnal:

    >> bokeh-server

    It will serve by default at http://localhost:5006/. You can pass an ip that
    the bokeh-server can listen to as well:

    >> bokeh-server --ip 0.0.0.0.

    This will allow you to see your progress from a different machine. This is
    nice if you browser and gpu server are on different places.

    In this example, we assume you are running a local server.
'''

# this live plots the training and validation loss
plot = Plot(fig_name='MNIST MLP example', url='default')

batch_size = 100
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

'''
We will visualize the weights of the first layer. Note that Grid2D assumes
each filter is in a different row, this is why we transpose W.
'''
grid = Grid2D(fig_name="First layer weights", url='default',
              W=model.layers[0].W.T) # TODO transpose W by default?

model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test), callbacks=[plot, grid])

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
