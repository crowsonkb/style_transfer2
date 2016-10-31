class SetImages:
    """A request from the app to the worker to set the image in a specific slot. Slots include
    'content', 'style', 'input', etc. An image parameter should be a NumPy array in HxWx3 layout
    with RGB channel order.

    If a slot is set to None, it is left alone, unless that would
    create inconsistency i.e. the content and input images must be the same size. In that case the
    None slot will be set to an array of zeros. Alternately you can set that slot to
    SetImages.RESAMPLE, which will direct the worker to resample the image to the given size.

    If reset_state is true, the iterate count will be reset back to zero and the optimizer's
    internal state will be cleared."""
    RESAMPLE = 1

    def __init__(self, size=None, input_image=None, content_image=None, style_image=None,
                 reset_state=False):
        self.size = size
        self.input_image = input_image
        self.content_image = content_image
        self.style_image = style_image
        self.reset_state = reset_state


class SetStepSize:
    """A request from the app to the worker to set the optimizer's step size (learning rate)."""
    default_adam = 10
    default_lbfgs = 1

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


class Shutdown:
    """Signals the receiving process to shut down."""
    pass


class StartIteration:
    """Signals the worker to start iteration."""
    pass


class PauseIteration:
    """Signals the worker to pause iteration."""
    pass


class Iterate:
    """A notification from the worker to the app that a new iterate has been produced. It contains
    the image as a NumPy float32 (or convertable-to-float32, since the receiver will call
    np.float32() on all arrays received) array in HxWx3 layout with RGB channel order. 'i' is the
    number of iterates produced since the start of iteration."""
    def __init__(self, image, loss, i):
        self.image = image
        self.loss = loss
        self.i = i
