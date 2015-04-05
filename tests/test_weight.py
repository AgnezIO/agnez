import numpy as np
from agnez.weight.image import grid2d


def test_grid2d():
    W = np.random.normal(size=(100, 100))
    A = grid2d(W)
    assert A.shape == (111, 111)
