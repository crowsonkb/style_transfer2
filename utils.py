import sys

import numpy as np
from PIL import Image
from scipy import ndimage

logging_format = '%(module)s: %(levelname)s: %(message)s'


def setup_exceptions(mode='Plain', color_scheme='Neutral'):
    try:
        from IPython.core import ultratb
        sys.excepthook = ultratb.AutoFormattedTB(mode=mode, color_scheme=color_scheme)
    except ImportError:
        pass

setup_exceptions()


def scales(size, min_size=1, factor=np.sqrt(2)):
    """Returns a list of scales that increase from min_size to size by a given factor."""
    size = np.float64(size)
    min_size = int(min_size)
    assert min_size >= 1

    sizes = [tuple(int(round(x)) for x in size)]
    while True:
        size /= factor
        size_int = tuple(int(round(x)) for x in size)
        if max(size_int) < min_size:
            break
        sizes.append(size_int)
    sizes.reverse()
    return sizes


def resize(arr, size, order=3):
    """Resamples a CxHxW NumPy float array to a different HxW shape."""
    arr = np.float32(arr)
    h, w = size
    hh, ww = arr.shape[1:]
    resized_arr = ndimage.zoom(arr, (1, h/hh, w/ww), order=order, mode='nearest')
    assert resized_arr.shape[1:] == size
    return resized_arr


def tv_norm(x, beta=2):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis and [3]."""
    x_diff, y_diff = np.zeros_like(x), np.zeros_like(x)
    x_diff[..., :, :-1] -= np.diff(x, axis=-1)
    y_diff[..., :-1, :] -= np.diff(x, axis=-2)
    grad_norm2 = x_diff**2 + y_diff**2 + 1e-8
    norm = np.sum(grad_norm2**(beta/2))
    dgrad_norm = (beta/2) * grad_norm2**(beta/2 - 1)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[..., :, 1:] -= x_diff[..., :, :-1]
    grad[..., 1:, :] -= y_diff[..., :-1, :]
    return norm, grad


def as_pil(arr):
    """Converts a NumPy HxWxC float array to a PIL image."""
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
