import inspect
import logging

import numpy as np

import optimizers

logger = logging.getLogger(__name__)


class Message:
    """The base class for the messages exchanged via PyZMQ's send_pyobj()."""
    debug = False

    def __repr__(self):
        def repr_value(v):
            if isinstance(v, np.ndarray):
                return '<ndarray, shape: %s, dtype: %s>' % (v.shape, v.dtype)
            return repr(v)

        args = ['%s=%s' % (k, repr_value(v)) for k, v in sorted(self.__dict__.items())]
        return self.__class__.__name__ + '(' + ', '.join(args) + ')'

    def _debug(self):
        if self.debug:
            frame = inspect.currentframe()
            try:
                caller = frame.f_back.f_back
                logger.debug('%s created on line %d of %s: %s', self.__class__.__name__,
                             caller.f_lineno, caller.f_code.co_filename, repr(self))
            finally:
                del frame


class Iterate(Message):
    """A notification from the worker to the app that a new iterate has been produced. It contains
    the image as a NumPy float32 (or convertable-to-float32, since the receiver will call
    np.float32() on all arrays received) array in HxWx3 layout with RGB channel order. 'i' is the
    number of iterates produced since the start of iteration."""
    def __init__(self, image, loss, i):
        self.image = image
        self.loss = loss
        self.i = i
        self._debug()


class PauseIteration(Message):
    """Signals the worker to pause iteration."""
    def __init__(self):
        self._debug()


class SetImages(Message):
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
        self._debug()


class SetOptimizer(Message):
    """A request from the app to the worker to set the optimizer type and optionally step size.
    Step sizes will be taken from the per-optimizer defaults in this message type if not
    specified."""
    classes = {'adam': optimizers.AdamOptimizer,
               'lbfgs': optimizers.LBFGSOptimizer}
    step_sizes = {'adam': 10, 'lbfgs': 1}

    def __init__(self, optimizer, step_size=None):
        self.optimizer = optimizer
        if not step_size:
            step_size = self.step_sizes[optimizer]
        self.step_size = step_size
        self._debug()


class SetWeights(Message):
    """A request from the app to the worker to set the loss weights for each combination of layer
    and loss type. Loss types can be divided into two categories: those which are only valid for
    layers not the input layer, and those which are only valid for the input layer.

    Args:
        weights: A dict of dicts of weights, arranged thusly: weights['content']['conv2_2'] is the
            weight for content loss on the layer conv2_2.
        params: A dict of weights which are only valid for the input layer, and miscellaneous
                options influencing their action i.e. exponents.
            i.e. params['tv'].
    """
    loss_names = ('content', 'style', 'deepdream')
    scalar_loss_names = ('tv', 'tv_power', 'p', 'p_power')

    def __init__(self, weights, params):
        self.weights = weights
        self.params = params
        self._debug()


class Shutdown(Message):
    """Signals the receiving process to shut down."""
    def __init__(self):
        self._debug()



class StartIteration(Message):
    """Signals the worker to start iteration."""
    def __init__(self):
        self._debug()



class WorkerReady(Message):
    """Signals the app that the worker is ready to receive messages and should be initialized."""
    def __init__(self, send_images=False):
        self.send_images = send_images
        self._debug()
