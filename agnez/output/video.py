import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import wraps

from .embedding import _prepare_fig_labels
from ..weight import grid2d


def _prepare_axis(subplot, data=None):
    ax = plt.subplot(subplot, aspect='equal')
    if data is not None:
        xymin = data.min()-data.std()/3
        xymax = data.max()+data.std()/3
        plt.xlim(xymin, xymax)
        plt.ylim(xymin, xymax)
    ax.axis('off')
    return ax


def animate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        make_frame, fig, fargs, video_length, ani_path = func(*args, **kwargs)
        ani = animation.FuncAnimation(fig, make_frame, frames=video_length, interval=100,
                                      fargs=fargs)
        ani.save(ani_path, writer='imagemagick', fps=5)
        return ani
    return wrapper


@animate
def timeseries2dvideo(data, labels, ani_path='ts2video.gif'):
    '''2d scatter plot video of times series embedding

    Parameters
    ----------
    data: `numpy.array`
        numpy array with dimensions (time, samples, 2)
    labels: `numpy.array`
        numpy vector with the label of each sample in data. `labels`
        must have the same number of elements as the second dimension
        of data
    ani_path: str
        path to save the animation
    '''
    labels, palette, fig = _prepare_fig_labels(data, labels)
    ax = _prepare_axis(111, data)
    t, b, d = data.shape
    data = data.transpose(1, 0, 2).reshape((t*b, d))
    sc = ax.scatter([], [])

    def make_frame(t, sc):
        pts = data[t]
        color = np.hstack([palette[labels[t].astype(np.int)], 1.])
        offsets = np.vstack([sc.get_offsets(), pts])
        sc.set_offsets(offsets)
        colors = np.vstack([sc.get_facecolors(), color])
        sc.set_facecolors(colors)
    return make_frame, fig, (sc,), data.shape[0], ani_path

    # TODO delete this
    # ani = animation.FuncAnimation(fig, make_frame, frames=data.shape[0], interval=100, fargs=(sc,))
    # ani.save(ani_path, writer='imagemagick', fps=10)
    # return ani


@animate
def video_embedding(video, embedding, labels, ani_path='video_ebd.gif'):
    '''2D scatter plot video of times series embedding along side
    its original image sequence.

    Parameters
    ----------
    video: 3D `numpy.array`
        array with image sequences with dimensions (frames, samples, dim)
    embedding: 3D `numpy.array`
        2D embedding of each video with dimensions (frames, samples, 2)
    labels: `numpy.array`
        numpy vector with the label of each sample in data. `labels`
        must have the same number of elements as the second dimension
        of data
    ani_path: str
        path to save the animation

    '''
    labels, palette, fig = _prepare_fig_labels(embedding, labels)
    ax2 = _prepare_axis(121, embedding)
    ax1 = _prepare_axis(122)
    sc = ax2.scatter([], [])

    t, b, d = embedding.shape
    embedding = embedding.transpose(1, 0, 2).reshape((t*b, d))
    t, b, d = video.shape
    video = video.transpose(1, 0, 2).reshape((t*b, d))
    dim = np.sqrt(d).astype('int')

    init_frame = video[0].reshape((dim, dim))
    vid = ax1.imshow(init_frame, cmap='gist_gray_r', vmin=video.min(), vmax=video.max())
    # plt.draw()

    def make_frame(t, sc, vid):
        pts = embedding[t]
        frame = video[t].reshape((dim, dim))
        color = np.hstack([palette[labels[t].astype(np.int)], 1.])
        offsets = np.vstack([sc.get_offsets(), pts])
        sc.set_offsets(offsets)
        colors = np.vstack([sc.get_facecolors(), color])
        sc.set_facecolors(colors)
        vid.set_data(frame)
        return sc, vid

    return make_frame, fig, (sc, vid), t*b, ani_path


@animate
def video_grid(video, ani_path='video_grid.gif', rescale=False):
    '''2D video grid for parallel visualization

    Parameters
    ----------
    video: 3D `numpy.array`
        array with image sequences with dimensions (frames, samples, dim)
    ani_path: str
        path to save the animation
    rescale: bool
        flag to rescale displayed images by grid2d

    '''
    fig = plt.figure()
    ax1 = _prepare_axis(111)
    t, b, d = video.shape

    grid = grid2d(video[:, 0, :])

    vid = ax1.imshow(grid, cmap='gray')
    # plt.draw()

    def make_frame(t, vid):
        grid = grid2d(video[t], rescale=rescale)
        vid.set_data(grid)
        return vid

    return make_frame, fig, (vid,), t, ani_path
