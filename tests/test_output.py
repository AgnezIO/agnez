import numpy as np

import matplotlib.pyplot as plt

from agnez import embedding2d, embedding2dplot


def test_embedding2d():
    data = np.random.normal((5, 8))
    labels = np.arange(5)
    ebd, mtd = embedding2d(data)
    assert ebd.shape == (5, 2)

    fig, ax, sc, txts = embedding2dplot(ebd, labels)
    assert isinstance(fig, plt.Figure)
