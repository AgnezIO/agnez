import numpy as np


def grid2d(X, example_width=False, display_cols=False, pad_row=1, pad_col=1):
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
        integer number of pixels between vertical neighbors
    pad_col: int
        integer number of pixels between horizontal neighbors

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
                newData = (X[curr_ex, :].reshape((example_height, example_width))).T/max_val
            except:
                raise ValueError("expected {}, got {}".format(X[curr_ex, :].shape), (example_height, example_width))
            display_array[i_inds, j_inds] = newData.flatten()
            curr_ex += 1
        if curr_ex >= m:
            break
    visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    return visual  # display_array


def pref_grid(above, bellow, num_preferred=10, abs_value=True, pad_col=5):
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
        maximum absolute activation. Else, uses only the maximum positve
        values.
    pad_col: int
        integer number of pixels between horizontal neighbors

    """
    idx = np.random.randint(above.shape[0], size=num_preferred)
    R = np.abs(above[idx]) if abs_value else above[idx]
    X = np.zeros((num_preferred**2, bellow.shape[1]))
    for i, w in enumerate(R):
        s = np.argsort(w)
        prefs = s[:-num_preferred-1:-1]
        first = i*num_preferred
        last = (i+1)*num_preferred
        X[first:last] = bellow[prefs]
    visual = grid2d(X, pad_row=1, pad_col=pad_col)
    return visual[:, pad_col-1:-pad_col+1]
