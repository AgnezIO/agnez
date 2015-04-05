import numpy as np
from past.utils import old_div


def grid2d(W, width=False, num_vectors=None):
    """
    Display weights as a 2D grid
    =============================

    This function assumes the input weights W is a matrix where each column
    is a projection vector. It them reshapes the those vectors to
    width x height and group them in a 2D grid.
    """
    if num_vectors is not None:
        W = W[:, :num_vectors]
    m, n = W.shape
    if not width:
        width = int(np.round(np.sqrt(n)))
    height = old_div(m, width)
    # Compute number of items to display
    display_cols = int(np.sqrt(m))
    display_rows = int(np.ceil(m/display_cols))
    pad = 1
    # Setup blank display
    display_array = -np.ones((pad+display_rows * (height+pad),
                              pad+display_cols * (width+pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = abs(W[curr_ex, :]).max()
            i_inds = width*[pad+j * (height+pad)+q for q in range(height)]
            j_inds = [pad+i * (width+pad)+q
                      for q in range(width)
                      for nn in range(height)]
            try:
                newData = (W[curr_ex, :].reshape((height, width))).T/max_val
            except:
                print W[curr_ex, :].shape
                print (height, width)
                raise
            display_array[i_inds, j_inds] = newData.flatten()
            curr_ex += 1
        if curr_ex >= m:
            break
    # Display the image
    # visual = (display_array - display_array.min()) / (display_array.max() - display_array.min())
    return display_array
