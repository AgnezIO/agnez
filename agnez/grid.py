from __future__ import division
import numpy as np
# from gaborfitting import *
import theano
import theano.tensor as T


def scale_norm(X):
    X = X - X.min()
    scale = (X.max() - X.min())
    return X / scale


def img_grid(X, rows_cols=None, rescale=True):
    """Image Grid: modified from jbornschein/draw

    Parameters:
    ===========
    X : np.array, images (samples, channels, height, width)
    rows_cols : list, grid dimensions (rows, cols)
    rescale : bool

    Returns:
    ========
    I : np.array, grid image
    """
    N, channels, height, width = X.shape

    if rows_cols is None:
        cols = np.ceil(np.sqrt(X.shape[0])).astype('int')
        rows = np.ceil(X.shape[0] / cols).astype('int')
    else:
        rows, cols = rows_cols

    total_height = rows * height + rows - 1
    total_width = cols * width + cols - 1

    if rescale:
        X = scale_norm(X)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if rescale:
            this = X[i]
        else:
            this = scale_norm(X[i])

        offset_y, offset_x = r*height+r, c*width+c
        I[0:channels, offset_y:(offset_y+height),
          offset_x:(offset_x+width)] = this

    I = (255*I).astype(np.uint8)
    if(channels == 1):
        out = I.reshape((total_height, total_width))
    else:
        out = np.dstack(I).astype(np.uint8)
    return out


def grid2d(X, example_width=False, display_cols=False, pad_row=1,
           pad_col=1, rescale=True):
    """Display weights in a nice grid

    This function assumes that each row of the X is an image weight to be
    resized to a square. After that it creates a 2D grid with with all
    the squares.

    Parameters
    ----------
    X : `numpy.array`
        array with each filter to be transformed to an image on the rows
    example_width: int
        defines the width of the images in the rows X if they are
        not square
    display_cols: bool
    pad_row: int
        integer number of pixels between up/down neighbors
    pad_col: int
        integer number of pixels between left/right neighbors

    Adapted from https://github.com/martinblom/py-sparse-filtering

    """
    m, n = X.shape
    if not example_width:
        example_width = int(np.round(np.sqrt(n)))
    example_height = n//example_width
    # Compute number of items to display
    if not display_cols:
        display_cols = int(np.sqrt(m))
    display_rows = int(np.ceil(m/display_cols))
    # Setup blank display
    display_array = -np.ones((pad_row+display_rows * (example_height+pad_row),
                              pad_col+display_cols * (example_width+pad_col)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = abs(X[curr_ex, :]).max()
            i_inds = example_width*[pad_row+j * (example_height+pad_row)+q for q in range(example_height)]
            j_inds = [pad_col+i * (example_width+pad_col)+q
                      for q in range(example_width)
                      for nn in range(example_height)]
            try:
                newData = (X[curr_ex, :].reshape((example_height,
                                                  example_width))).T/max_val
            except:
                raise ValueError("expected {}, got {}".format(X[curr_ex,:].shape), (example_height, example_width))
            display_array[i_inds, j_inds] = newData.flatten()
            curr_ex += 1
        if curr_ex >= m:
            break
    visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    visual = np.nan_to_num(visual)
    ret = visual if rescale else display_array
    ret = (255*ret).astype(np.uint8)
    return ret


def pref_grid(above, bellow, num_preferred=9, abs_value=True, pad_row=5):
    """Display the weights that the layer above prefers on the layer below

    This function looks for the `num_preferred` larger values on the layer
    `above` and get their indexes. Those indexes are used to retrieve the
    preferred weights on the layer `bellow`. After all, those preferred
    vectors are organized with `meth`:grid2d.

    Parameters
    ----------
    above : `numpy.array`
        matrix with each filter to be transformed to an image on the rows
    bellow : `numpy.array`
        matrix with each filter to be transformed to an image on the rows
    num_preferred: int
        number of preferred weights to be plotted
    abs_value: bool
        if True chooses the preferred as the weights associated with
        maximum absolute activation. Else, uses only the maximum (positve)
        values.
    pad_row: int
        integer number of pixels between up/down neighbors

    """
    # idx = np.random.randint(above.shape[0], size=num_preferred)
    R = np.abs(above) if abs_value else above
    X = np.zeros((num_preferred**2, bellow.shape[1]))
    for i, w in enumerate(R):
        s = np.argsort(w)[::-1]
        prefs = s[:num_preferred]
        first = i*num_preferred
        last = (i+1)*num_preferred
        X[first:last] = bellow[prefs]
    visual = grid2d(X, pad_col=1, pad_row=pad_row)
    return visual[pad_row-1:-pad_row+1, :]


class DeepPref():
    """Similar do pref_grid but for deep networks.
    Checks what are the weights in layers[0] that layers[-1] prefers.

    Parameters
    ----------
    model: `keras.models.Sequential`
    layer: int, observed layer
    num_preferred: int
        number of preferred weights to be plotted
    abs_value: bool
        if True chooses the preferred as the weights associated with
        maximum absolute activation. Else, uses only the maximum (positve)
        values.
    pad_row: int
        integer number of pixels between horizontal neighbors

    """
    def __init__(self, model, layer, num_preferred=10, abs_value=True,
                 pad_row=5, sum_preferences=False):
        self.model = model
        self.layer = layer
        self.num_preferred = num_preferred
        self.abs_value = abs_value
        self.pad_row = pad_row
        self.sum_preferences = sum_preferences
        X = model.get_input()
        Y = model.layers[layer].get_output()
        if self.sum_preferences:
            Y = T.nnet.softmax(Y)
        self.F = theano.function([X], Y, allow_input_downcast=True)
        num_weights_out = model.layers[layer].W.get_value().shape[1]
        self.idx = np.random.randint(num_weights_out,
                                     size=num_preferred)

    def get_pref(self):
        W = self.model.layers[0].W.get_value().T
        Y = self.F(W)
        R = np.abs(Y[:, self.idx]) if self.abs_value else Y[:, self.idx]
        if self.sum_preferences:
            X = np.zeros((self.num_preferred, W.shape[1]))
        else:
            X = np.zeros((self.num_preferred**2, W.shape[1]))
        for i, w in enumerate(R.T):
            s = np.argsort(w)
            prefs = s[:-self.num_preferred-1:-1]
            first = i*self.num_preferred
            last = (i+1)*self.num_preferred
            if self.sum_preferences:
                X[i] = (W[prefs]).mean(axis=0)
            else:
                X[first:last] = W[prefs]
        visual = grid2d(X, pad_col=1, pad_row=self.pad_row)
        return visual[self.pad_row-1:-self.pad_row+1, :]
