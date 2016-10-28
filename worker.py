#!/usr/bin/env python3

"""The worker module for the under-development web app."""

import configparser
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import zmq

import messages
import utils

MODULE_DIR = Path(__file__).parent.resolve()

ctx = zmq.Context()
logger = logging.getLogger(__name__)


class CaffeModel:
    """A Caffe neural network model."""
    models_path = MODULE_DIR / 'models'
    model_path = models_path / 'vgg19.prototxt'
    weights_path = models_path / 'vgg19.caffemodel'
    mean = np.float32((123.68, 116.779, 103.939)).reshape((3, 1, 1))

    def __init__(self, gpu):
        # Set environment variables before the first import of caffe, then import it
        logger.debug('Initializing Caffe.')
        os.environ['GLOG_minloglevel'] = '1'
        if gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            import caffe  # pylint: disable=import-error
            caffe.set_mode_gpu()
        else:
            import caffe  # pylint: disable=import-error
            caffe.set_mode_cpu()

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


class StyleTransfer:
    def __init__(self, model):
        self.model = model
        self.is_running = False
        self.i = 0
        self.input = None
        self.features = None
        self.grams = None
        self.step_size = 1
        # TODO: receive actual weights from the app
        loss_types = ('content', 'style', 'tv')
        layers = self.model.layers()
        arr = np.ones((len(layers), len(loss_types)), np.float32)
        self.weights = pd.DataFrame(arr, layers, loss_types, np.float32)

    def set_input(self, image):
        self.input = self.model.preprocess(image)

    def set_content(self, image):
        image = self.model.preprocess(image)
        self.features = self.model.forward(image)

    def set_style(self, image):
        image = self.model.preprocess(image)
        features = self.model.forward(image)
        self.grams = {}
        for layer, feat in features.items():
            _, n, mh, mw = feat.shape
            feat = feat.reshape((n, mh * mw))
            self.grams[layer] = np.dot(feat, feat.T) / np.float32(feat.size)

    def step(self):
        self.i += 1
        # Get list of layers to provide gradients to
        nonzeros = self.weights != 0
        layers = self.weights.index[nonzeros.sum(axis=1) != 0]

        # Compute the loss and gradient at each of those layers
        current_feats = self.model.forward(self.input)
        loss = 0
        diffs = {}
        for layer in layers:
            # Content gradient
            c_grad = current_feats[layer] - self.features[layer]
            c_grad *= 2 / c_grad.size
            loss += self.weights.content[layer] * np.sum(c_grad**2) / c_grad.size
            diffs[layer] = self.weights.content[layer] * c_grad

        # Get the combined gradient via backpropagation
        grad = self.model.backward(diffs)

        # Add the total-variation loss and gradient
        tv_loss, tv_grad = utils.tv_norm(self.input / 255)
        loss += self.weights.tv['data'] * tv_loss
        grad += self.weights.tv['data'] * tv_grad

        # Take a gradient descent step
        self.input -= self.step_size * grad

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
                    new_msg = messages.Iterate(image, loss, self.transfer.i)
                    self.sock_out.send_pyobj(new_msg)
                continue
            msg = self.sock_in.recv_pyobj()
            self.process_message(msg)

    def process_message(self, msg):
        if isinstance(msg, messages.SetImage):
            if msg.slot == 'input':
                self.transfer.set_input(msg.image)
            elif msg.slot == 'content':
                self.transfer.set_content(msg.image)
            elif msg.slot == 'style':
                self.transfer.set_style(msg.image)
            else:
                logger.warning('Invalid message received.')

        elif isinstance(msg, messages.StartIteration):
            self.transfer.is_running = True

        else:
            logger.warning('Invalid message received.')

def main():
    """The main function."""
    cp = configparser.ConfigParser()
    cp.read(str(MODULE_DIR / 'config.ini'))
    config = cp['DEFAULT']

    if 'caffe_python_module' in config:
        sys.path.append(config['caffe_python_module'])

    logging.basicConfig(level=logging.DEBUG, format=utils.logging_format)
    logging.captureWarnings(True)

    Worker(config).run()


if __name__ == '__main__':
    main()
