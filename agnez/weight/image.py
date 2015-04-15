import numpy as np

from agnez import Artist
from agnez.weights import as_matrix
from agnez.weights.grid2d import grid2d


class Grid2D(Artist):
    def __init__(self, *args, **kwargs):
        super(Grid2D, self).__init__(*args, **kwargs)

    def art(self):
        product = [grid2d(as_matrix(W)) for W in self.Weights]
