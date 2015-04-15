import numpy as np


def grid2d(X, example_width=False, display_cols=True):
    """
    Display 2D data in a nice grid
    ==============================

    Displays 2D data stored in X in a nice grid. It returns the
    figure handle and the displayed array.

    Borrowed from https://github.com/martinblom/py-sparse-filtering
    """
    # compute rows, cols
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
    # Display the image
    visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    return visual  # display_array
