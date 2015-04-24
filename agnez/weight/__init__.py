import numpy as np


def grid2d(X, example_width=False, display_cols=False):
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

    Borrowed from https://github.com/martinblom/py-sparse-filtering
    """
    m, n = X.shape
    if not example_width:
        example_width = int(np.round(np.sqrt(n)))
    example_height = n//example_width
    # Compute number of items to display
    if not display_cols:
        display_cols = int(np.sqrt(m))
    display_rows = int(np.ceil(m/display_cols))
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+display_rows * (example_height+pad),
                              pad+display_cols * (example_width+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = abs(X[curr_ex, :]).max()
            i_inds = example_width*[pad+j * (example_height+pad)+q for q in range(example_height)]
            j_inds = [pad+i * (example_width+pad)+q
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
