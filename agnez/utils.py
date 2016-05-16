import io
import base64
import matplotlib.pyplot as plt
try:
  from IPython.display import HTML
except:
  print("can't import video_at_notebook")


def to_01c(img):
    """Make sure we get row, col, channels"""
    if len(img) == 2:
        return img
    else:
        if img.shape[0] == 3:
            return img.transpose(1, 2, 0)
        else:
            return img



def to_c01(img):
    """Make sure we get channels, row, col"""
    if len(img) == 2:
        return img
    else:
        if img.shape[2] == 3:
            return img.transpose(2, 0, 1)
        else:
            return img


def imshow(img):
    if len(img.shape) == 2:  # gray
        plt.imshow(img, cmap="gray")
    else:  # color
        if img.shape[0] == 3:
            plt.imshow(img.transpose(1, 2, 0))
        else:
            plt.imshow(img)


def video_at_notebook(video_path):
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
