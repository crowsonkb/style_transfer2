#!/usr/bin/env python3

"""The worker module for Style Transfer."""

import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import zmq

from messages import *
import optimizers
import utils

utils.setup_exceptions()

MODULE_DIR = Path(__file__).parent.resolve()

ctx = zmq.Context()
logger = logging.getLogger('worker')

caffe_import_msg = '''
ImportError: Caffe was not found in PYTHONPATH. Please edit config.ini to
contain the line "caffe_path = <path to compiled Caffe>."'''


class CaffeModel:
    """A Caffe neural network model."""
    mean = np.float32((123.68, 116.779, 103.939)).reshape((3, 1, 1))

    def __init__(self, prototxt, caffemodel, gpu=-1):
        self.prototxt = str(prototxt)
        self.caffemodel = str(caffemodel)

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

        self.reload_net()
        logger.info('Caffe initialized.')

    def reload_net(self):
        """Recreates the neural network in its initial state."""
        import caffe
        self.net = caffe.Net(self.prototxt, 1, weights=self.caffemodel)

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
        # if np.prod(self.net.blobs['data'].data.shape) > np.prod(image.shape):
        #     self.reload_net()
        self.net.blobs['data'].reshape(*image.shape)
        # logger.debug('req shape: %s, shape: %s', image.shape, self.net.blobs['data'].data.shape)
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
        self.is_starting = False
        self.t = 0
        self.input = None
        self.content = None
        self.features = None
        self.grams = None
        weights_shape = (len(self.model.layers()), len(SetWeights.loss_names))
        self.weights = pd.DataFrame(
            np.ones(weights_shape), self.model.layers(), SetWeights.loss_names, np.float32)
        self.params = {w: 1 for w in SetWeights.scalar_loss_names}
        self.optimizer = None
        self.optimizer_cls = optimizers.LBFGSOptimizer
        self.step_size = SetOptimizer.step_sizes['lbfgs']
        self.norms = {k: {} for k in 'cds'}
        self.traces = []

    def objective_changed(self):
        if self.optimizer is not None:
            self.optimizer.objective_changed()

    def pause(self):
        self.is_running = False
        self.is_starting = False

    def resample_input(self, size):
        if self.input is not None and self.optimizer is not None:
            self.input = self.optimizer.resample(size)
        else:
            self.input = np.zeros((1, 3) + size, np.float32)
        self._start()
        self.objective_changed()

    def resample_content(self, size):
        if self.content is not None:
            self.content = utils.resample_nchw(self.content, size)
        else:
            self.content = np.zeros((1, 3) + size, np.float32)
        features = self.model.forward(self.content)
        self.features = {k: v.copy() for k, v in features.items()}
        self._start()
        self.objective_changed()

    def reset(self):
        self.norms = {k: {} for k in self.norms}
        self.t = 0
        self.optimizer = self.optimizer_cls(self.input, self.opfunc, step_size=self.step_size)

    def start(self):
        self.is_starting = True
        self._start()
        return self.is_running

    def _start(self):
        if self.is_starting:
            if self.input is not None and self.content is not None and self.grams:
                if self.input.shape == self.content.shape:
                    if self.optimizer is None:
                        self.reset()
                    self.is_starting = False
                    self.is_running = True

    def set_input(self, image):
        image = self.model.preprocess(image)
        if self.input is not None and self.input.shape == image.shape:
            self.input[:] = image
            self.objective_changed()
        elif self.optimizer is not None:
            self.input = self.optimizer.resample(None, new_x=image)
            self._start()
        else:
            self.input = image
            self.reset()
            self._start()

    def set_content(self, image):
        self.content = self.model.preprocess(image)
        features = self.model.forward(self.content)
        self.features = {k: v.copy() for k, v in features.items()}
        self._start()
        self.objective_changed()

    def set_style(self, image):
        image = self.model.preprocess(image)
        features = self.model.forward(image)
        self.grams = {}
        for layer, feat in features.items():
            self.grams[layer] = gram_matrix(feat)
        self._start()
        self.objective_changed()

    def set_step_size(self, step_size):
        """Sets the optimizer's step size."""
        self.step_size = step_size
        if self.optimizer is not None:
            self.optimizer.step_size = step_size

    def set_weights(self, weights, params):
        self.weights = pd.DataFrame.from_dict(weights, dtype=np.float32)
        self.params = params
        self.objective_changed()

    def opfunc(self, x, return_grad=True):
        """Calculates the objective function and its gradient."""
        # Get list of layers to provide gradients to
        nonzeros = abs(self.weights) > 1e-15
        layers = self.weights.index[abs(nonzeros.sum(axis=1)) > 1e-15]
        t = utils.Trace()

        # Compute the loss and gradient at each of those layers
        current_feats = self.model.forward(x, layers)
        loss = 0
        diffs = {}
        for layer in layers:
            w = self.weights
            cw, sw, dw = w['content'][layer], w['style'][layer], w['deepdream'][layer]
            diffs[layer] = np.zeros_like(current_feats[layer])
            n_ = ['%s_%s_%s' % (layer, lt, lg) for lt in 'dsc' for lg in ('grad', 'loss')].pop

            # Content gradient
            if abs(cw) > 1e-15:
                c_grad = current_feats[layer] - self.features[layer]
                c_grad *= 2 / c_grad.size
                if layer not in self.norms['c']:
                    self.norms['c'][layer] = np.sqrt(np.mean(c_grad**2))
                loss += t(n_(), cw * np.mean(c_grad**2) / self.norms['c'][layer])
                diffs[layer] += t.rms(n_(), cw * c_grad / self.norms['c'][layer])

            # Style gradient
            if abs(sw) > 1e-15:
                _, n, mh, mw = current_feats[layer].shape
                gram_diff = gram_matrix(current_feats[layer]) - self.grams[layer]
                feat = current_feats[layer].reshape((n, mh * mw))
                s_grad = (2 / gram_diff.size) * np.dot(gram_diff, feat).reshape((1, n, mh, mw))
                if layer not in self.norms['s']:
                    self.norms['s'][layer] = np.sqrt(np.mean(s_grad**2))
                loss += t(n_(), sw * np.mean(gram_diff**2) / self.norms['s'][layer])
                t.rms(n_(), sw / self.norms['s'][layer] * s_grad)
                utils.axpy(sw / self.norms['s'][layer], s_grad, diffs[layer])

            # Deep Dream gradient
            if abs(dw) > 1e-15:
                d_grad = -2 * current_feats[layer] / current_feats[layer].size
                if layer not in self.norms['d']:
                    self.norms['d'][layer] = np.sqrt(np.mean(d_grad**2))
                loss -= t(n_(), dw * np.mean(current_feats[layer]**2) / self.norms['d'][layer])
                diffs[layer] += t.rms(n_(), dw * d_grad / self.norms['d'][layer])

        n_ = ['%s_%s' % (lt, lg) for lg in ('grad', 'loss') for lt in 'pt'].pop
        # Get the total variation loss and gradient
        tv_loss, tv_grad = utils.tv_norm(x / 255, self.params['tv_power'])
        loss += t(n_(), self.params['tv'] * tv_loss)

        p_loss, p_grad = utils.p_norm(x / 255, self.params['p_power'])
        loss += t(n_(), self.params['p'] * p_loss)

        if not return_grad:
            self.traces.append(t)
            return t('loss', loss)

        # Get the combined gradient
        grad = t.rms('sc_grad', self.model.backward(diffs).copy())
        grad += t.rms(n_(), self.params['tv'] * tv_grad)
        grad += t.rms(n_(), self.params['p'] * p_grad)

        t('time', time.perf_counter())
        self.traces.append(t)
        return t('loss', loss), t.rms('grad', grad)

    def step(self):
        """Returns the next iterate and the current value of the loss function."""
        self.t += 1
        x, loss = self.optimizer.step()
        # logger.debug('step %d, %s', self.t, self.traces[-1])
        t = self.traces[-1]
        t('t', self.t)
        return self.model.deprocess(t.rms('step', x)), loss

    def write_trace(self, filename):
        df = pd.DataFrame(t.data for t in self.traces)
        df.index.name = 'step'
        df.to_csv(filename)


class Worker:
    """The worker main class."""
    def __init__(self, config):
        self.sock_in = ctx.socket(zmq.PULL)
        self.sock_out = ctx.socket(zmq.PUSH)
        self.sock_in.bind(config['worker_socket'])
        self.sock_out.connect(config['app_socket'])

        prototxt = MODULE_DIR / config['prototxt']
        caffemodel = MODULE_DIR / config['caffemodel']
        gpu = config.getint('gpu', fallback=-1)
        model = CaffeModel(prototxt, caffemodel, gpu)

        self.transfer = StyleTransfer(model)
        self.sock_out.send_pyobj(WorkerReady())

    def run(self):
        try:
            should_stop = False
            while not should_stop:
                if self.transfer.is_running:
                    try:
                        while True:
                            msg = self.sock_in.recv_pyobj(zmq.NOBLOCK)
                            if self.process_message(msg):
                                should_stop = True
                                break
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
        finally:
            self.transfer.write_trace('trace.csv')

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
            self.transfer.optimizer_cls = SetOptimizer.classes[msg.optimizer]
            self.transfer.set_step_size(msg.step_size)
            if not isinstance(self.transfer.optimizer, self.transfer.optimizer_cls):
                self.transfer.reset()

        elif isinstance(msg, SetWeights):
            self.transfer.set_weights(msg.weights, msg.params)

        elif isinstance(msg, Shutdown):
            return True

        elif isinstance(msg, StartIteration):
            if not self.transfer.start():
                self.sock_out.send_pyobj(WorkerReady(send_images=True))

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

    debug = config.getboolean('debug', False)
    if debug:
        utils.setup_exceptions(mode='Context')
    utils.setup_logging(debug)

    Worker(config).run()

    logger.info('Shutting down worker process.')


if __name__ == '__main__':
    main()
