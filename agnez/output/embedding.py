import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.decomposition import PCA
import matplotlib.animation as animation


def embedding2d(data, train_data=None, method=None):
    '''2D embedding for visualization

    '''
    if method is None:
        method = PCA(n_components=2)
    if train_data is None:
        ebd = method.fit_transform(data)
    else:
        method.fit(train_data)
        ebd = method.transform(data)
    return ebd, method


def embedding2dplot(data, labels):
    '''2D embedding visualization.

    Modified from:
    https://beta.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

    '''
    # We choose a color palette with seaborn.
    max_label = labels.max()
    palette = np.array(sns.color_palette("hls", max_label+1))
    # We create a scatter plot.
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(data[:, 0], data[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(data[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return fig, ax, sc, txts


def timeseries2d(data, train_data=None, method=None):
    '''2D embedding for time-series

    data: `numpy.array`
        array with axes [time, batch, dim]
    labels: `numpy.array`
        array with axes [batch, ]

    '''
    t, b, d = data.shape
    data = data.reshape((t*b, d))
    ebd, mtd = embedding2d(data, train_data, method)
    ebd = ebd.reshape((t, b, -1))
    return ebd, mtd


def timeseries2dplot(data, labels):
    '''2D scatter plot of time series using embeding2d

    '''
    t, b, d = data.shape
    data = data.reshape((t*b, d))
    labels = labels[np.newaxis].repeat(t, axis=0)
    labels = labels.flatten()
    ebd, mtd = embedding2d(data)
    fig, ax, sc, txts = embedding2dplot(ebd, labels)
    return fig, ax, sc, txts


def timeseries2dvideo(data, labels, ani_path='ts2video.gif'):
    '''2D scatter plot video of times series embedding

    '''
    # We choose a color palette with seaborn.
    max_label = labels.max()
    palette = np.array(sns.color_palette("hls", max_label+1))
    # We create a scatter plot.

    # We add the labels for each digit.
    t, b, d = data.shape
    data = data.transpose(1, 0, 2).reshape((t*b, d))
    labels = labels[np.newaxis].repeat(t, axis=0).transpose(1, 0)
    labels = labels.flatten()

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    xymin = data.min()-data.std()/3
    xymax = data.max()+data.std()/3
    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    ax.axis('off')

    def make_frame(t, fig, ax):
        pts = data[t]
        color = palette[labels[t].astype(np.int)]
        ax.scatter(pts[0], pts[1], c=color)
        return ax

    ani = animation.FuncAnimation(fig, make_frame, frames=data.shape[0], interval=100, fargs=(fig, ax))
    ani.save(ani_path, writer='imagemagick', fps=20)
    return ani
