class SetImage:
    """A request from the app to the worker to set the image in a specific slot. Slots include
    'content', 'style', 'input', etc. The image parameter should be a NumPy array in HxWx3 layout
    with RGB channel order."""
    slot_names = ('input', 'content', 'style')

    def __init__(self, slot, image):
        self.slot = slot
        self.image = image


class SetStepSize:
    """A request from the app to the worker to set the optimizer's step size (learning rate)."""
    default = 1

    def __init__(self, step_size):
        self.step_size = step_size


class SetWeights:
    """A request from the app to the worker to set the loss weights for each combination of layer
    and loss type. Loss types can be divided into two categories: those which are only valid for
    layers not the input layer, and those which are only valid for the input layer.

    Args:
        weights: A dict of dicts of weights, arranged thusly: weights['content']['conv2_2'] is the
            weight for content loss on the layer conv2_2.
        scalar_weights: A dict of weights which are only valid for the input layer.
            i.e. scalar_weights['tv'].
    """
    loss_names = ('content', 'style')
    scalar_loss_names = ('tv',)

    def __init__(self, weights, scalar_weights):
        self.weights = weights
        self.scalar_weights = scalar_weights


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
