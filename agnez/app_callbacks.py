import os
import requests
import json
import mpld3
import numpy as np
import matplotlib.pyplot as plt
app_url = 'http://localhost:3000/api/v1/values'

from keras.callbacks import Callback, History
from keras import backend as K
try:
    from imgurpython import ImgurClient
except:
    print "imgrupython not installed, can't use SendImgur"

from .grid import img_grid


class Sender(Callback):
    def __init__(self, name, description, position, cell_type,
                 app_url=app_url):
        super(Sender, self).__init__()
        self.app_url = app_url
        self.name = name
        self.description = description
        self.position = position
        self.cell_type = cell_type
        r = requests.post(app_url, json={
            'name': name, 'type': self.cell_type, 'value': '',
            'pos': '', 'description': self.description})
        self.app_url += '/' + str(json.loads(r.text)['_id'])


class SendImgur(Sender):
    def __init__(self, generate_img, name='img', description='', position=1,
                 app_url=app_url):
        super(SendImgur, self).__init__(name, description, position,
                                        'img', app_url)
        self.client_secret = os.environ['IMGUR_SECRET']
        self.client_id = os.environ['IMGUR_ID']
        self.client = ImgurClient(self.client_id, self.client_secret)
        self.generate_img = generate_img

    def on_epoch_end(self, epoch=None, logs={}):
        img = self.generate_img()
        res = self.client.upload_from_path(img)
        return requests.patch(self.app_url, json={
            'name': self.name, 'type': 'img', 'value': res['link'],
            'pos': self.position, 'description': self.description})


class SendFigHTML(Sender):
    def __init__(self, generate_plot, name='plot', description='', position=0,
                 app_url=app_url):
        super(SendFigHTML, self).__init__(name, description, position,
                                          'html', app_url)
        self.generate_plot = generate_plot

    def on_epoch_end(self, epoch=None, logs={}):
        fig = self.generate_plot()
        html = mpld3.fig_to_html(fig)
        return requests.patch(self.app_url, json={
            'name': self.name, 'type': 'html', 'value': html,
            'pos': self.position, 'description': self.description})


class LossPlot(Sender, History):
    def __init__(
            self, name='Loss plot', position=0, app_url=app_url,
            description='train (blue) and validation (green) learning curves'):
        super(LossPlot, self).__init__(name, description, position,
                                       'html', app_url)
        self.train_values = []
        self.valid_values = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_values.append(self.totals['loss']/self.seen)
        self.valid_values.append(logs['val_loss'])

        fig = plt.figure(figsize=(8, 5))
        plt.plot(self.train_values)
        plt.plot(self.valid_values)
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        html = mpld3.fig_to_html(fig)
        plt.close(fig)

        requests.patch(self.app_url, json={
            'name': self.name, 'type': 'html', 'value': html,
            'pos': self.position, 'description': self.description})


class VisualizeConvWeights(Sender):
    def __init__(self, weights, static_path, name='Conv weights',
                 position=1, app_url=app_url,
                 description='weights of convolutional layer'):
        super(VisualizeConvWeights, self).__init__(name, description, position,
                                                   'img', app_url)
        self.weights = weights
        self.static_path = static_path

    def on_epoch_end(self, epoch, logs={}):
        W = np.asarray(K.eval(self.weights))
        I = img_grid(W)
        fig = plt.figure(figsize=(8, 5))
        plt.imshow(I)
        plt.savefig(os.path.join(self.static_path, 'images',
                                 self.name+'.png'))
        # html = mpld3.fig_to_html(fig)
        plt.close(fig)

        requests.patch(self.app_url, json={
            'name': self.name, 'type': 'img',
            'value': os.path.join('./images', self.name+'.png'),
            'pos': self.position, 'description': self.description})
