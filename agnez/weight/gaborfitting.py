import numpy as np
import theano
import theano.tensor as tensor

from keras.layers.core import Layer
from keras.models import Sequential
from keras import initializations
from keras.optimizers import RMSprop

floatX = theano.config.floatX


class GaborFit(Layer):
    def __init__(self, input_dim, output_dim, octave=True):
        super(GaborFit, self).__init__()
        init0 = initializations.get('zero')
        init1 = initializations.get('uniform')
        xydim = np.sqrt(output_dim)
        x, y = np.meshgrid((np.linspace(-1, 1, xydim),)*2)
        self.x = theano.shared(x.ravel().astype(floatX))
        self.y = theano.shared(y.ravel().astype(floatX))
        self.x0 = init0((1,))
        self.y0 = init0((1,))
        self.theta = init0((input_dim,))
        self.omega = init1((input_dim,))
        self.input = tensor.matrix()
        if octave:
            self.kappa = 2.5
        else:
            self.kappa = np.pi

    def _outter(self, t1, t2):
        return tensor.tensordot(t1, t2, axes=([], []))

    def get_output(self, train=False):
        rnorm = self.omega / np.sqrt(2*np.pi)*self.kappa
        val = - self.omage**2 / (8 * self.kappa**2)
        dir1 = 4 * (self._outter(self.x, tensor.cos(self.theta)) +
                    self._outter(self.y, tensor.sin(self.theta)))**2
        dir2 = (-self._outter(self.x, tensor.sin(self.theta)) +
                self._outter(self.y, tensor.cos(self.theta)))**2
        ex = 1j * (self.omage * self._outter(tensor.cos(self.theta), self.x) +
                   self.omage * self._outter(tensor.sin(self.theta), self.y))
        output = rnorm * tensor.exp(val * (dir1 + dir2)) * (tensor.exp(ex)
                                                            - tensor.exp(-self.kappa**2 / 2))
        return output


def gaborfit(W, nb_epoch=1000):
    model = Sequential()
    model.add(GaborFit(W.shape[0], W.shape[1]))
    model.compile(loss='mse', optimizer=RMSprop())
    model.fit(W, W, batch_size=len(W), nb_epoch=nb_epoch, show_accuracy=False,
              verbose=0)
