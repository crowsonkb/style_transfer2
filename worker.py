#!/usr/bin/env python3

"""The worker module for Style Transfer."""

import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import blas
import zmq

from messages import *
import optimizers
import utils

utils.setup_exceptions()

MODULE_DIR = Path(__file__).parent.resolve()

ctx = zmq.Context()
logger = logging.getLogger(__name__)

caffe_import_msg = '''
ImportError: Caffe was not found in PYTHONPATH. Please edit config.ini to
contain the line "caffe_path = <path to compiled Caffe>."'''


class CaffeModel:
    """A Caffe neural network model."""
    models_path = MODULE_DIR / 'models'
    model_path = models_path / 'vgg19.prototxt'
    weights_path = models_path / 'vgg19.caffemodel'
    mean = np.float32((123.68, 116.779, 103.939)).reshape((3, 1, 1))

    def __init__(self, gpu=-1):
        # Set environment variables before the first import of caffe, then import it
        logger.info('Initializing Caffe.')
        os.environ['GLOG_minloglevel'] = '1'
        try:
            if gpu >= 0:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
                import caffe  # pylint: disable=import-error
                caffe.set_mode_gpu()
            else:
                import caffe  # pylint: disable=import-error
                caffe.set_mode_cpu()
        except ImportError:
            print(caffe_import_msg, file=sys.stderr)
            sys.exit(2)

        self.net = caffe.Net(str(self.model_path), 1, weights=str(self.weights_path))
        logger.info('Caffe initialized.')

    def preprocess(self, image):
        """Preprocesses an input image for use in the network."""
        arr = np.float32(image).transpose((2, 0, 1)) - self.mean
        return np.ascontiguousarray(arr[None, ::1])

    def deprocess(self, image):
        """Reverses the action of preprocess()."""
        arr = (image.squeeze()[::1] + self.mean).transpose((1, 2, 0))
        return arr

    def layers(self):
        """Returns the layer names of the network."""
        return [layer for layer in self.net.blobs.keys() if layer.find('_split_') == -1]

    def forward(self, image, layers=None):
        """Runs the network forward, returning an OrderedDict of feature maps of the given
        preprocessed image. They will be overwritten on the next call to forward()."""
        if layers is None:
            layers = self.layers()
        self.net.blobs['data'].reshape(*image.shape)
        return self.net.forward(data=image, blobs=list(layers))

    def backward(self, diffs):
        """Runs the network backward, adding each layer gradient in diffs to that layer. Returns
        the gradient at the input layer. It will be overwritten on the next call to backward()."""
        # Put the layers to visit in the order we will visit them
        layers = [layer for layer in reversed(self.layers()) if layer in diffs.keys()]

        # Clear the prior state of each layer
        for layer in layers:
            self.net.blobs[layer].diff[:] = 0

        # Visit each layer
        for i, layer in enumerate(layers):
            self.net.blobs[layer].diff[:] += diffs[layer]
            if i < len(layers) - 1:
                self.net.backward(start=layer, end=layers[i+1])
            else:
                self.net.backward(start=layer)

        return self.net.blobs['data'].diff


def gram_matrix(x):
    """Computes the Gram matrix of a feature map."""
    n, c, h, w = x.shape
    assert n == 1
    x = x.reshape((c, h * w))
    return np.dot(x, x.T) / np.float32(x.size)


class StyleTransfer:
    """The class which performs image stylization. StyleTransfer calculates and manages state
    related to an objective function (StyleTransfer.opfunc()) which it minimizes using an Optimizer
    instance."""
    def __init__(self, model):
        self.model = model
        self.is_running = False
        self.t = 0
        self.input = None
        self.content = None
        self.features = None
        self.grams = None
        weights_shape = (len(self.model.layers()), len(SetWeights.loss_names))
        self.weights = pd.DataFrame(
            np.ones(weights_shape), self.model.layers(), SetWeights.loss_names, np.float32)
        self.scalar_weights = {w: 1 for w in SetWeights.scalar_loss_names}
        self.optimizer = None
        self.optimizer_cls = optimizers.LBFGSOptimizer
        self.c_grad_norms = {}
        self.s_grad_norms = {}

    def objective_changed(self):
        self.c_grad_norms = {}
        self.s_grad_norms = {}
        if self.optimizer is not None:
            self.optimizer.objective_changed()

    def pause(self):
        self.is_running = False

    def resample_input(self, size):
        if self.input is not None and self.optimizer is not None:
            self.input = self.optimizer.resample(size)
        else:
            self.input = np.zeros((1, 3) + size, np.float32)
        self.objective_changed()

    def resample_content(self, size):
        if self.content is not None:
            self.content = utils.resize(self.content, size)
        else:
            self.content = np.zeros((1, 3) + size, np.float32)
        features = self.model.forward(self.content)
        self.features = {k: v.copy() for k, v in features.items()}
        self.objective_changed()

    def reset(self):
        self.c_grad_norms = {}
        self.s_grad_norms = {}
        self.t = 0
        assert self.input is not None, "No input image provided yet."
        self.optimizer = self.optimizer_cls(self.input, self.opfunc)

    def start(self):
        if self.optimizer is None:
            self.reset()
        self.is_running = True

    def set_input(self, image):
        image = self.model.preprocess(image)
        if self.input is not None and self.input.shape == image.shape:
            self.input[:] = image
            self.objective_changed()
        elif self.optimizer is not None:
            self.optimizer.resample(None, new_x=image)
        else:
            self.input = image
            self.reset()

    def set_content(self, image):
        self.content = self.model.preprocess(image)
        features = self.model.forward(self.content)
        self.features = {k: v.copy() for k, v in features.items()}
        self.objective_changed()

    def set_style(self, image):
        image = self.model.preprocess(image)
        features = self.model.forward(image)
        self.grams = {}
        for layer, feat in features.items():
            self.grams[layer] = gram_matrix(feat)
        self.objective_changed()

    # def set_step_size(self, step_size):
    #     """Sets the optimizer's step size."""
    #     self.step_size = step_size
    #     if self.optimizer is not None:
    #         self.optimizer.step_size = step_size

    def set_weights(self, weights, scalar_weights):
        self.weights = pd.DataFrame.from_dict(weights, dtype=np.float32)
        self.scalar_weights = scalar_weights
        self.objective_changed()

    def opfunc(self, x, return_grad=True):
        """Calculates the objective function and its gradient."""
        # Get list of layers to provide gradients to
        nonzeros = abs(self.weights) > 1e-15
        layers = self.weights.index[abs(nonzeros.sum(axis=1)) > 1e-15]

        # Compute the loss and gradient at each of those layers
        current_feats = self.model.forward(x, layers)
        loss = 0
        diffs = {}
        for layer in layers:
            cw, sw = self.weights['content'][layer], self.weights['style'][layer]
            diffs[layer] = np.zeros_like(self.features[layer])

            # Content gradient
            if abs(cw) > 1e-15:
                c_grad = current_feats[layer] - self.features[layer]
                c_grad *= 2 / c_grad.size
                if layer not in self.c_grad_norms:
                    self.c_grad_norms[layer] = np.sqrt(np.mean(c_grad**2))
                loss += cw * np.mean(c_grad**2) / self.c_grad_norms[layer]
                diffs[layer] += cw * c_grad / self.c_grad_norms[layer]

            # Style gradient
            if abs(sw) > 1e-15:
                _, n, mh, mw = current_feats[layer].shape
                gram_diff = gram_matrix(current_feats[layer]) - self.grams[layer]
                feat = current_feats[layer].reshape((n, mh * mw))
                s_grad = (2 / feat.size) * np.dot(gram_diff, feat).reshape((1, n, mh, mw))
                if layer not in self.s_grad_norms:
                    self.s_grad_norms[layer] = np.sqrt(np.mean(s_grad**2))
                loss += sw * np.mean(gram_diff**2) / self.s_grad_norms[layer]
                utils.axpy(sw / self.s_grad_norms[layer], s_grad, diffs[layer])

        # Get the total variation loss and gradient
        tv_loss, tv_grad = utils.tv_norm(x / 255)
        loss += self.scalar_weights['tv'] * tv_loss

        if not return_grad:
            return loss

        # Get the combined gradient via backpropagation
        grad = self.model.backward(diffs).copy()
        grad += self.scalar_weights['tv'] * tv_grad

        return loss, grad

    def step(self):
        """Take a gradient descent step."""
        self.t += 1
        x, loss = self.optimizer.step()
        return self.model.deprocess(x), loss


class Worker:
    """The worker main class."""
    def __init__(self, config):
        self.sock_in = ctx.socket(zmq.PULL)
        self.sock_out = ctx.socket(zmq.PUSH)
        self.sock_in.bind(config['worker_socket'])
        self.sock_out.connect(config['app_socket'])
        gpu = config.getint('gpu', fallback=-1)
        model = CaffeModel(gpu)
        self.transfer = StyleTransfer(model)

    def run(self):
        try:
            while True:
                if self.transfer.is_running:
                    try:
                        msg = self.sock_in.recv_pyobj(zmq.NOBLOCK)
                        self.process_message(msg)
                    except zmq.ZMQError:
                        image, loss = self.transfer.step()
                        new_msg = Iterate(image, loss, self.transfer.t)
                        self.sock_out.send_pyobj(new_msg)
                    continue
                msg = self.sock_in.recv_pyobj()
                if self.process_message(msg):
                    break
        except KeyboardInterrupt:
            self.sock_out.send_pyobj(Shutdown())

    def process_message(self, msg):
        def is_image(obj):
            return obj is not None and not isinstance(obj, int)

        if isinstance(msg, SetImages):
            if is_image(msg.input_image):
                self.transfer.set_input(msg.input_image)
            elif msg.input_image == SetImages.RESAMPLE:
                self.transfer.resample_input(msg.size)

            if is_image(msg.content_image):
                self.transfer.set_content(msg.content_image)
            elif msg.content_image == SetImages.RESAMPLE:
                self.transfer.resample_content(msg.size)

            if is_image(msg.style_image):
                self.transfer.set_style(msg.style_image)

            if msg.reset_state:
                self.transfer.reset()

        elif isinstance(msg, SetOptimizer):
            classes = dict(adam=optimizers.AdamOptimizer,
                           lbfgs=optimizers.LBFGSOptimizer)
            self.transfer.optimizer_cls = classes[msg.optimizer]
            if not isinstance(self.transfer.optimizer, classes[msg.optimizer]):
                self.transfer.reset()

        # elif isinstance(msg, SetStepSize):
        #     self.transfer.set_step_size(msg.step_size)

        elif isinstance(msg, SetWeights):
            self.transfer.set_weights(msg.weights, msg.scalar_weights)

        elif isinstance(msg, Shutdown):
            return True

        elif isinstance(msg, StartIteration):
            self.transfer.start()

        elif isinstance(msg, PauseIteration):
            self.transfer.pause()

        else:
            logger.error('Invalid message received over ZeroMQ.')

        return False


def main():
    """The main function."""
    config = utils.read_config()
    if 'caffe_path' in config:
        sys.path.append(config['caffe_path'] + '/python')

    utils.setup_logging()

    Worker(config).run()
    logger.info('Shutting down worker process.')


if __name__ == '__main__':
    main()
