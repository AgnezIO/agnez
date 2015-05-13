import numpy as np

import matplotlib.pyplot as plt

from agnez import embedding2d, embedding2dplot, timeseries2d, timeseries2dplot


def test_embedding2d():
    data = np.random.normal(0, 1, (5, 8))
    labels = np.arange(5)
    ebd, mtd = embedding2d(data)
    assert ebd.shape == (5, 2)

    fig, ax, sc, txts = embedding2dplot(ebd, labels)
    assert isinstance(fig, plt.Figure)


def test_timesseries2d():
    data = np.random.normal(0, 1, (3, 5, 8))
    labels = np.arange(5)
    ebd, mtd = timeseries2d(data)
    assert ebd.shape == (3, 5, 2)

    fig, ax, sc, txts = timeseries2dplot(ebd, labels)
    assert isinstance(fig, plt.Figure)
