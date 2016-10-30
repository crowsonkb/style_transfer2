from contextlib import contextmanager
import logging
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


@contextmanager
def profile():
    from accelerate import profiler
    prof = profiler.Profile()
    prof.enable()
    yield
    prof.disable()
    prof.print_stats(1)
    prof.clear()


def setup_logging():
    fmt = '%(asctime)s.%(msecs)03d %(filename)s %(levelname)s: %(message)s'
    datefmt = '%H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    logging.captureWarnings(True)



def setup_exceptions(mode='Plain', color_scheme='Neutral'):
    try:
        from IPython.core import ultratb
        sys.excepthook = ultratb.AutoFormattedTB(mode=mode, color_scheme=color_scheme)
    except ImportError:
        pass


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


def resize(arr, size, order=1):
    """Resamples an NxCxHxW NumPy float array to a different HxW shape."""
    arr = np.float32(arr)
    h, w = size
    hh, ww = arr.shape[2:]
    resized_arr = ndimage.zoom(arr, (1, 1, h/hh, w/ww), order=order)
    assert resized_arr.shape[2:] == size
    return resized_arr


def resize_to_fit(image, size, scale_up=False):
    """Resizes a PIL image to fit into a size-by-size square."""
    size = int(round(size))
    w, h = image.size
    if not scale_up and max(w, h) <= size:
        return image
    new_w, new_h = w, h
    if w > h:
        new_w = size
        new_h = int(round(size * h/w))
    else:
        new_h = size
        new_w = int(round(size * w/h))
    return image.resize((new_w, new_h), Image.LANCZOS)


def roll_by_1(arr, shift, axis):
    """Rolls a 4D array in-place by a shift of one element. Axes 2 and 3 only."""
    if axis == 2:
        if shift == -1:
            line = arr[:, :, 0, :].copy()
            arr[:, :, :-1, :] = arr[:, :, 1:, :]
            arr[:, :, -1, :] = line
        elif shift == 1:
            line = arr[:, :, -1, :].copy()
            arr[:, :, 1:, :] = arr[:, :, :-1, :]
            arr[:, :, 0, :] = line
    elif axis == 3:
        if shift == -1:
            line = arr[:, :, :, 0].copy()
            arr[:, :, :, :-1] = arr[:, :, :, 1:]
            arr[:, :, :, -1] = line
        elif shift == 1:
            line = arr[:, :, :, -1].copy()
            arr[:, :, :, 1:] = arr[:, :, :, :-1]
            arr[:, :, :, 0] = line
    else:
        raise ValueError('Unsupported shift or axis')
    return arr


def tv_norm(x, beta=2):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis and [3]."""
    x_diff = x - roll_by_1(x.copy(), -1, axis=3)
    y_diff = x - roll_by_1(x.copy(), -1, axis=2)
    grad_norm2 = x_diff**2 + y_diff**2 + 1e-8
    loss = np.sum(grad_norm2**(beta/2))
    dgrad_norm = (beta/2) * grad_norm2**(beta/2 - 1)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad -= roll_by_1(dx_diff, 1, axis=3)
    grad -= roll_by_1(dy_diff, 1, axis=2)
    return loss, grad


def as_pil(arr):
    """Converts a NumPy HxWxC float array to a PIL image."""
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
