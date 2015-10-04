'''This provides visualization tools for Keras.'''
from . import grid2d, DeepPref, video_grid

from theano import function
from keras.callbacks import Callback
from bokeh.plotting import (cursession, figure, output_server,
                            push, show)


class BokehCallback(Callback):
    def __init__(self, name, fig_title, url):
        """
        fig_title: Figure Title
        url : str, optional
            Url of the bokeh-server. Ex: when starting the bokeh-server with
            ``bokeh-server --ip 0.0.0.0`` at ``alice``, server_url should be
            ``http://alice:5006``. When not specified the default configured
            by ``bokeh_server`` in ``.blocksrc`` will be used. Defaults to
            ``http://localhost:5006/``.

        Reference: mila-udem/blocks-extras
        """
        Callback.__init__(self)
        self.name = name
        self.fig_title = fig_title
        self.plots = []
        output_server(name, url=url)
        cursession().publish()

    def get_image(self):
        raise NotImplemented

    def on_epoch_end(self, epoch, logs={}):
        I = self.get_image()
        if not hasattr(self, 'fig'):
            self.fig = figure(title=self.fig_title,
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


class Plot(BokehCallback):
    # WIP
    # TODO:
    #   -[ ] Decide API for choosing channels to plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    '''
    Inspired by https://github.com/mila-udem/blocks-extras/blob/master/blocks/extras/extensions/plot.py

    '''
    def __init__(self, name='experiment', fig_title='Cost functions',
                 url='default'):
        BokehCallback.__init__(self, name, fig_title, url)
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
            self.fig = figure(title=self.fig_title)
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
    W: weight matrix to visualize, each filter should go into a row,
       filters sizes must be square

    See also BokehCallback `on_batch_end` method

    '''
    def __init__(self, W, num_weights=100, name='experiment',
                 fig_title='grid', url='default',):
        BokehCallback.__init__(self, name, fig_title, url)
        self.W = function([], W)
        self.num_weights = num_weights

    def get_image(self):
        W = self.W()
        return grid2d(W[:self.num_weights])


class PreferedInput(BokehCallback):
    '''
    model: Keras Sequential model
    layer: int (>= 1) value of the desired layer to visualize

    NOTE: This method calculates the prefered first layer weights of a upper
          hidden layer. It simply checks what are the strongest connections.

    See also BokehCallback `on_batch_end` method

    '''
    def __init__(self, model, layer, name='experiment', fig_title='pref_grid',
                 url='default', sum_preferences=False):
        BokehCallback.__init__(self, name, fig_title, url)
        self.model = model
        self.sum_preferences = sum_preferences
        self.deep_pref = DeepPref(model, layer, sum_preferences=sum_preferences)

    def get_image(self):
        return self.deep_pref.get_pref()


class SaveGif(Callback):
    def __init__(self, filepath, X, func, how_often=10, display=None):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.how_often = how_often
        self.display = display
        self.X = X
        self.func = func

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.how_often == 0:
            rec = self.func(self.X)
            _ = video_grid(rec.transpose(1, 0, 2), self.filepath)
            if self.display is not None:
                self.display.clear_output(wait=True)
        print('Saved gif.')
