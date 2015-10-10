import os
import requests
import json
import mpld3

from keras.callbacks import Callback
from imgurpython import ImgurClient


class Sender(Callback):
    def __init__(self, name, position,
                 app_url='http://localhost:3000/values'):
        super(Sender, self).__init__()
        self.app_url = app_url
        self.name = name
        self.position = position
        r = requests.post(app_url, json={
            'name': name, 'type': '', 'value': '',
            'pos': ''})
        self.app_url += '/' + str(json.loads(r.text)['id'])


class SendImgur(Sender):
    def __init__(self, generate_img, name='img', position=1,
                 app_url='http://localhost:3000/values'):
        super(SendImgur, self).__init__(name, position, app_url)
        self.client_secret = os.environ['IMGUR_SECRET']
        self.client_id = os.environ['IMGUR_ID']
        self.client = ImgurClient(self.client_id, self.client_secret)
        self.generate_img = generate_img

    def on_epoch_end(self, epoch=None, logs={}):
        img = self.generate_img()
        res = self.client.upload_from_path(img)
        return requests.patch(self.app_url, json={
            'name': self.name, 'type': 'img', 'value': res['link'],
            'pos': self.position})


class SendFigHTML(Sender):
    def __init__(self, generate_plot, name='plot', position=0,
                 app_url='http://localhost:3000/values'):
        super(SendFigHTML, self).__init__(name, position, app_url)
        self.generate_plot = generate_plot

    def on_epoch_end(self, epoch=None, logs={}):
        fig = self.generate_plot()
        html = mpld3.fig_to_html(fig)
        return requests.patch(self.app_url, json={
            'name': self.name, 'type': 'html', 'value': html,
            'pos': self.position})
