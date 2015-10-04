

def fool(model, data, desired):
    u"""Starting from `data`, this method modifies the input forcing the
    the output of the `model` to be as close as possible to `desired`.

    In the case of a classification model and a mesleading desired,
    we have the adversarial examples for neural networks [Szeg14]_

    Parameters
    ----------
    model : `Blocks.Model` instance
        This is basically a `Theano` graph with an apply method

    .. [Szeg14] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever,
        Joan Bruna, Dumitru Erhan, Ian Goodfellow, Rob Fergus,
        Intriguing properties of neural networks, ICLR 2014.

    """
