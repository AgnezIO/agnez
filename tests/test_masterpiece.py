import os
import numpy

from agnez import Masterpiece
from agnez.weight import Grid2D


def test_masterpiece():
    W = numpy.random.normal(size=(100, 100))
    model = (W,)
    artists = [Grid2D, ]
    mp = Masterpiece(artists=artists, model=model)
    mp.expose('test')
    file_path = os.path.dirname(os.path.realpath(__file__))
    assert os.path.exists(os.path.join(file_path, 'test.ipynb'))
