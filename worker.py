#!/usr/bin/env python3

"""The worker module for Style Transfer."""

import configparser
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import zmq

from messages import *
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
        logger.debug('Initializing Caffe.')
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
        logger.debug('Caffe initialized.')

    def preprocess(self, image):
        """Preprocesses an input image for use in the network."""
        arr = np.float32(image).transpose((2, 0, 1)) - self.mean
        return np.ascontiguousarray(arr[None, ::1])

    def deprocess(self, image):
        """Reverses the action of preprocess()."""
        arr = (image.squeeze()[::1] + self.mean).transpose((1, 2, 0))
        return np.ascontiguousarray(arr)

    def layers(self):
        """Returns the layer names of the network."""
        return [layer for layer in self.net.blobs.keys() if layer.find('_split_') == -1]

    def forward(self, image):
        """Runs the network forward, returning an OrderedDict of feature maps of the given
        preprocessed image. They will be overwritten on the next call to forward()."""
        self.net.blobs['data'].reshape(*image.shape)
        return self.net.forward(data=image, blobs=self.layers())

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


class AdamOptimizer:
    """An optimizer using the Adam algorithm for stochastic approximation. Given the parameters
    array x, it takes scaled gradient descent steps when supplied with x's gradient. The step_size
    value controls the maximum amount a parameter may change per step. b1 and b2 are Adam momentum
    parameters which should not need to be changed from the default."""
    def __init__(self, x, step_size=SetStepSize.default, b1=0.9, b2=0.999):
        self.x = x
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.t = 0
        self.g1 = np.zeros_like(x)
        self.g2 = np.zeros_like(x)

    def step(self, grad):
        """Takes a scaled gradient descent step given x's gradient. Updates x in place, and returns
        the new value of x."""
        self.t += 1

        self.g1[:] = self.b1*self.g1 + (1-self.b1)*grad
        self.g2[:] = self.b2*self.g2 + (1-self.b2)*grad**2
        ss = self.step_size * np.sqrt(1-self.b2**self.t) / (1-self.b1**self.t)

        self.x -= ss * self.g1 / (np.sqrt(self.g2) + 1e-8)
        return self.x

    def resize(self, size):
        """Resamples the optimizer's internal state on the last two axes to a new HxW size."""
        self.g1 = utils.resize(self.g1, size)
        self.g2 = np.maximum(utils.resize(self.g2, size, order=1), 0)
        self.x = utils.resize(self.x, size)
        return self.x


def gram_matrix(x):
    """Computes the Gram matrix of a feature map."""
    n, c, h, w = x.shape
    assert n == 1
    x = x.reshape((c, h * w))
    return np.dot(x, x.T) / np.float32(x.size)


class StyleTransfer:
    def __init__(self, model):
        self.model = model
        self.is_running = False
        self.input = None
        self.content = None
        self.features = None
        self.grams = None
        weights_shape = (len(self.model.layers()), len(SetWeights.loss_names))
        self.weights = pd.DataFrame(
            np.ones(weights_shape), self.model.layers(), SetWeights.loss_names, np.float32)
        self.scalar_weights = {w: 1 for w in SetWeights.scalar_loss_names}
        self.optimizer = None
        self.c_grad_norms = {}
        self.s_grad_norms = {}

    def set_input(self, image):
        self.input = self.model.preprocess(image)
        self.optimizer = AdamOptimizer(self.input)
        if self.content is None or self.input.shape != self.content.shape:
            self.content = np.zeros_like(self.input)

    def set_content(self, image):
        self.content = self.model.preprocess(image)
        if self.input is None or self.input.shape != self.content.shape:
            self.input = np.zeros_like(self.content)
        features = self.model.forward(self.content)
        self.features = {k: v.copy() for k, v in features.items()}

    def set_style(self, image):
        image = self.model.preprocess(image)
        features = self.model.forward(image)
        self.grams = {}
        for layer, feat in features.items():
            self.grams[layer] = gram_matrix(feat)

    def set_step_size(self, step_size):
        self.optimizer.step_size = step_size

    def set_weights(self, weights, scalar_weights):
        self.weights = pd.DataFrame.from_dict(weights, dtype=np.float32)
        self.scalar_weights = scalar_weights

    def step(self):
        # Get list of layers to provide gradients to
        nonzeros = abs(self.weights) > 1e-15
        layers = self.weights.index[abs(nonzeros.sum(axis=1)) > 1e-15]

        # Compute the loss and gradient at each of those layers
        current_feats = self.model.forward(self.input)
        loss = 0
        diffs = {}
        for layer in layers:
            cw, sw = self.weights['content'][layer], self.weights['style'][layer]
            diffs[layer] = np.zeros_like(self.features[layer])

            # Content gradient
            if abs(cw) > 1e-15:
                c_grad = current_feats[layer] - self.features[layer]
                c_grad *= 2 / c_grad.size
                loss += cw * np.mean(c_grad**2)
                if layer not in self.c_grad_norms:
                    self.c_grad_norms[layer] = np.sqrt(np.mean(c_grad**2))
                diffs[layer] += cw * c_grad / self.c_grad_norms[layer]

            # Style gradient
            if abs(sw) > 1e-15:
                _, n, mh, mw = current_feats[layer].shape
                gram_diff = gram_matrix(current_feats[layer]) - self.grams[layer]
                feat = current_feats[layer].reshape((n, mh * mw))
                s_grad = 2 * np.dot(gram_diff, feat).reshape((1, n, mh, mw)) / feat.size
                if layer not in self.s_grad_norms:
                    self.s_grad_norms[layer] = np.sqrt(np.mean(s_grad**2))
                loss += sw * np.mean(gram_diff**2) / self.s_grad_norms[layer]
                diffs[layer] += sw * s_grad / self.s_grad_norms[layer]

        # Get the combined gradient via backpropagation
        grad = self.model.backward(diffs)

        # Add the total variation loss and gradient
        tv_loss, tv_grad = utils.tv_norm(self.input / 255)
        loss += self.scalar_weights['tv'] * tv_loss
        grad += self.scalar_weights['tv'] * tv_grad

        # Take a gradient descent step
        self.optimizer.step(grad)

        return self.model.deprocess(self.input), loss


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
        while True:
            if self.transfer.is_running:
                try:
                    msg = self.sock_in.recv_pyobj(zmq.NOBLOCK)
                    self.process_message(msg)
                except zmq.ZMQError:
                    image, loss = self.transfer.step()
                    new_msg = Iterate(image, np.sqrt(loss), self.transfer.optimizer.t)
                    self.sock_out.send_pyobj(new_msg)
                continue
            msg = self.sock_in.recv_pyobj()
            self.process_message(msg)

    def process_message(self, msg):
        if isinstance(msg, SetImage):
            if msg.slot == 'input':
                self.transfer.set_input(msg.image)
                self.sock_out.send_pyobj(Iterate(msg.image, np.nan, 0))
            elif msg.slot == 'content':
                self.transfer.set_content(msg.image)
            elif msg.slot == 'style':
                self.transfer.set_style(msg.image)
            else:
                logger.warning('Invalid message received.')

        elif isinstance(msg, SetStepSize):
            self.transfer.set_step_size(msg.step_size)

        elif isinstance(msg, SetWeights):
            self.transfer.set_weights(msg.weights, msg.scalar_weights)

        elif isinstance(msg, StartIteration):
            self.transfer.is_running = True

        else:
            logger.warning('Invalid message received.')

def main():
    """The main function."""
    config = utils.read_config()
    if 'caffe_path' in config:
        sys.path.append(config['caffe_path'] + '/python')

    utils.setup_logging()
    Worker(config).run()


if __name__ == '__main__':
    main()
