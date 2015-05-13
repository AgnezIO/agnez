import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.decomposition import PCA


def embedding2d(data, train_data=None, method=PCA(n_components=2)):
    '''2D embedding for visualization

    '''
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
        xtext, ytext = np.median(data[data == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return fig, ax, sc, txts
