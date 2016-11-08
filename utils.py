"""Various functions which must be used by both the app and its worker process."""

import argparse
import cProfile
import configparser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import logging
import os
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image
from scipy.linalg import blas

import messages

MODULE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = MODULE_DIR / 'config.ini'
CONFIG_PATH_NON_GIT = MODULE_DIR / 'config_non_git.ini'


# pylint: disable=no-member
def dot(x, y):
    """Returns the dot product of two float32 arrays of the same shape."""
    if x.shape != y.shape:
        raise ValueError('Sizes do not match: x=%s y=%s' % (x.shape, y.shape))
    x, y = x.ravel(), y.ravel()
    return blas.sdot(x, y)


# pylint: disable=no-member
def axpy(a, x, y):
    """Sets y = a*x + y for float a and float32 arrays x, y and returns y."""
    if x.shape != y.shape:
        raise ValueError('Sizes do not match: x=%s y=%s' % (x.shape, y.shape))
    x_, y_ = x.ravel(), y.ravel()
    y_ = blas.saxpy(x_, y_, a=a).reshape(y.shape)
    if y is not y_:
        y[:] = y_
    return y


class DecayingMean:
    """An exponentially weighted decaying mean with initialization bias correction. When called,
    returns the current mean. When called with a parameter, decays the mean and adds the parameter
    to it. If called while empty, returns NaN."""
    def __init__(self, decay=0.9):
        self.mean = 0
        self.decay = decay
        self.items = 0

    def __call__(self, item=None):
        if item is not None:
            self.mean = self.decay*self.mean + (1-self.decay)*item
            self.items += 1
        if self.items == 0:
            return self.mean
        return self.mean / (1 - self.decay**self.items)

    def clear(self):
        """Resets the decaying mean to empty."""
        self.mean = 0
        self.items = 0


@contextmanager
def profile():
    """A context manager which prints a profile for the time when execution was in its context."""
    prof = cProfile.Profile()
    prof.enable()
    yield
    prof.disable()
    prof.print_stats(1)
    prof.clear()


@contextmanager
def line_profile(items):
    """A context manager which prints a line-by-line profile for the given functions, modules, or
    module names while execution is in its context.

    Example:

    with line_profile(__name__, Class.some_function, some_module):
        do_something()
    """
    from line_profiler import LineProfiler
    prof = LineProfiler()
    for item in items:
        if inspect.isfunction(item):
            prof.add_function(item)
        elif inspect.ismodule(item):
            prof.add_module(item)
        elif isinstance(item, str):
            prof.add_module(sys.modules[str])
        else:
            raise TypeError('Inputs must be functions, modules, or module names')
    prof.enable()
    yield
    prof.disable()
    prof.print_stats()


def parse_args(desc=''):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', nargs='?', help='the config file')
    parser.add_argument('--debug', '-d', action='count', help='debug')
    args = parser.parse_args()
    if not args.debug:
        args.debug = 0
    return args


def read_config(args):
    """Returns a dict-like object consisting of key-value pairs from the configuration file."""
    cp = configparser.ConfigParser()
    configs = [str(CONFIG_PATH), str(CONFIG_PATH_NON_GIT)]
    if args.config:
        configs.append(args.config)
    cp.read(configs)
    return cp['DEFAULT']


def _resample(a, b, hw, method):
    b[:] = Image.fromarray(a).resize((hw[1], hw[0]), method)


def resample_hwc(a, hw, method=Image.LANCZOS):
    """Resamples an image array in HxWxC format to a new HxW size. The interpolation is performed
    in floating point and the result dtype is numpy.float32."""
    a = np.float32(a)
    ch = a.shape[-1]
    b = np.zeros((hw[0], hw[1], ch), np.float32)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = [ex.submit(_resample, a[:, :, i], b[:, :, i], hw, method) for i in range(ch)]
        _ = [fut.result() for fut in futs]

    return b


def resample_nchw(a, hw, method=Image.LANCZOS):
    """Resamples an image array in NxCxHxW format to a new HxW size. The interpolation is performed
    in floating point and the result dtype is numpy.float32."""
    a = np.float32(a)
    n, ch = a.shape[:2]
    b = np.zeros((n, ch, hw[0], hw[1]), np.float32)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = [[ex.submit(_resample, a[i, j], b[i, j], hw, method) for j in range(ch)]
                for i in range(n)]
        _ = [[fut.result() for fut in lst] for lst in futs]

    return b


def setup_exceptions(mode='Plain', color_scheme='Neutral'):
    """If it is present, uses IPython's ultratb to make exceptions more readable."""
    try:
        from IPython.core import ultratb
        sys.excepthook = ultratb.AutoFormattedTB(mode=mode, color_scheme=color_scheme)
    except ImportError:
        pass


def setup_logging(debug=0):
    """Sets the logging configuration for the current process."""
    fmt = '%(asctime)s.%(msecs)03d %(process)d %(name)s %(levelname)s: %(message)s'
    datefmt = '%H:%M:%S'
    if debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    if debug <= 1:
        logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    else:
        messages.Message.debug = True
    logging.captureWarnings(True)


def scales(size, min_size=1, factor=np.sqrt(2)):
    """Returns a list of scales that increase from min_size to size by a given factor."""
    size = np.float64(size)
    min_size = int(min_size)
    assert min_size >= 1

    sizes = [tuple(int(round(x)) for x in size)]
    while True:
        size /= factor
        size_int = tuple(int(round(x)) for x in size)
        if max(size_int) < min_size or min(size_int) < 1:
            break
        sizes.append(size_int)
    sizes.reverse()
    return sizes


def fit_into_square(current_size, size, scale_up=False):
    """Determines the aspect-preserving size that fits into a size-by-size square."""
    size = int(round(size))
    w, h = current_size
    if not scale_up and max(w, h) <= size:
        return current_size
    new_w, new_h = w, h
    if w > h:
        new_w = size
        new_h = int(round(size * h/w))
    else:
        new_h = size
        new_w = int(round(size * w/h))
    return (new_w, new_h)


def resize_to_fit(image, size, scale_up=True):
    """Resizes a PIL image to fit into a size-by-size square."""
    new_size = fit_into_square(image.size, size, scale_up)
    return image.resize(new_size, Image.LANCZOS)


def roll_by_one(arr, shift, axis):
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


class Trace:
    """A convenience class for recording snapshots of internal values."""
    def __init__(self):
        self.data = OrderedDict()

    def __call__(self, name, expr):
        while name in self.data:
            name += '_'
        if isinstance(expr, np.floating):
            self.data[name] = float(expr)
        elif isinstance(expr, np.integer):
            self.data[name] = int(expr)
        elif isinstance(expr, np.generic):
            warnings.warn('Did not convert NumPy scalar to Python scalar, may not be pickleable',
                          RuntimeWarning)
            self.data[name] = expr
        else:
            self.data[name] = expr
        return expr

    def __str__(self):
        return ', '.join('%s: %g' % item for item in self.data.items())

    def rms(self, name, expr):
        self(name, np.sqrt(np.mean(expr**2)))
        return expr


def tv_norm(x, beta=2):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    x_diff = x - roll_by_one(x.copy(), -1, axis=3)
    y_diff = x - roll_by_one(x.copy(), -1, axis=2)
    grad_norm2 = x_diff**2 + y_diff**2 + 1e-8
    norm = np.sum(grad_norm2**(beta/2))
    dgrad_norm = (beta/2) * grad_norm2**(beta/2 - 1)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad -= roll_by_one(dx_diff, 1, axis=3)
    grad -= roll_by_one(dy_diff, 1, axis=2)
    return norm, grad


def p_norm(x, p=2):
    """Computes 1/p of the p-norm to the p power and its gradient. From jcjohnson/cnn-vis."""
    norm = np.sum(abs(x)**p) / p
    grad = np.sign(x) * abs(x)**(p-1)
    return norm, grad


def as_pil(arr):
    """Converts a NumPy HxWxC float array to a PIL image."""
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
