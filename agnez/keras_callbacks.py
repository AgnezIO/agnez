'''This provides visualization tools for Keras.'''

import theano

from . import grid2d

from keras.callbacks import Callback
from bokeh.plotting import (cursession, figure, output_server,
                            push, show)


class BokehCallback(Callback):
    def __init__(self, fig_name, url):
        """
        fig_name: Figure Title
        url : str, optional
            Url of the bokeh-server. Ex: when starting the bokeh-server with
            ``bokeh-server --ip 0.0.0.0`` at ``alice``, server_url should be
            ``http://alice:5006``. When not specified the default configured
            by ``bokeh_server`` in ``.blocksrc`` will be used. Defaults to
            ``http://localhost:5006/``.
        """
        Callback.__init__(self)
        self.fig_name = fig_name
        self.plots = []
        output_server(fig_name, url=url)


class Plot(BokehCallback):
    # WIP
    # TODO:
    #   -[ ] Decide API for choosing channels to plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    '''
    Inspired by https://github.com/mila-udem/blocks-extras/blob/master/blocks/extras/extensions/plot.py

    '''
    def __init__(self, fig_name='training', url='default'):
        BokehCallback.__init__(self, fig_name, url)
        self.totals = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for v in ['loss']:
            if v in self.totals:
                self.totals[v] += logs.get(v) * batch_size
            else:
                self.totals[v] = logs.get(v) * batch_size

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self, 'fig'):
            self.fig = figure(title=self.fig_name)
            for i, v in enumerate(['loss', 'val_loss']):
                if v == 'loss':
                    L = self.totals[v] / self.seen
                else:
                    L = logs.get(v)
                self.fig.line([epoch], [L], legend=v,
                              name=v, line_width=2,
                              line_color=self.colors[i % len(self.colors)])
                renderer = self.fig.select({'name': v})
                self.plots.append(renderer[0].data_source)
            show(self.fig)
        else:
            for i, v in enumerate(['loss', 'val_loss']):
                if v == 'loss':
                    L = self.totals[v] / self.seen
                else:
                    L = logs.get(v)
                self.plots[i].data['y'].append(L)
                self.plots[i].data['x'].append(epoch)
        cursession().store_objects(self.plots[i])
        push()


class Grid2D(BokehCallback):
    '''
    Depends on agnez.grid2D

    W: weight to be visualize, each filter should go into a row, filters sizes
       must be square
    '''
    def __init__(self, W, fig_name='grid', url='default',):
        BokehCallback.__init__(self, fig_name, url)
        self.W = W
        self.F = theano.function([], W)

    def on_epoch_end(self, epoch, logs={}):
        I = grid2d(self.F())
        if not hasattr(self, 'fig'):
            self.fig = figure(title=self.fig_name,
                              x_range=[0, 1], y_range=[0, 1])
            self.fig.image(image=[I], x=[0], y=[0], dw=[1], dh=[1],
                           name='weight')
            renderer = self.fig.select({'name': 'weight'})
            self.plots.append(renderer[0].data_source)
            show(self.fig)
        else:
            self.plots[0].data['image'] = [I]
        cursession().store_objects(self.plots[0])
        push()
