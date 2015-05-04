import numpy as np

from agnez.inputs import image_sequence


def test_image_sequence():
    I = np.eye(28)[np.newaxis].repeat(10, axis=0)
    R = image_sequence(I.reshape(10, 28*28), shape=(28, 28))
    assert R.shape == (28, 28*10)
