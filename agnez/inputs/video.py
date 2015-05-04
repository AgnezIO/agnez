

def image_sequence(X, shape):
    '''Image Sequence converts a matrix with different examples in each
    row to a sequence of resized image

    Paramters
    ---------
    X: 2D `numpy.array`
    Matrix with flatten examples on each row

    shape: list
    list with the shape to resize the flatten elements in X

    '''
    X = X.reshape((-1,)+shape)
    X = X.transpose(1, 2, 0)
    return X
