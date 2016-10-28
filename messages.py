class SetImage:
    """A request from the app to the worker to set the image in a specific slot. Slots include
    'content', 'style', 'input', etc. The image parameter should be a NumPy array in HxWx3 layout
    with RGB channel order."""
    def __init__(self, slot, image):
        self.slot = slot
        self.image = image


class StartIteration:
    """Signals the worker to start iteration."""
    pass


# class PauseIteration:
#     """Signals the worker to pause iteration."""
#     pass


class Iterate:
    """A notification from the worker to the app that a new iterate has been produced. It contains
    the image as a NumPy float32 array in HxWx3 layout with RGB channel order. 'i' is the number of
    iterates produced since the start of iteration."""
    def __init__(self, image, loss, i):
        self.image = image
        self.loss = loss
        self.i = i
