#!/usr/bin/env python3

"""The main module for a web application implementing neural style transfer using Caffe. See
"A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576)."""

# pylint: disable=redefined-outer-name

import asyncio
import binascii
import io
import logging
import json
from pathlib import Path
import subprocess
import time

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
import numpy as np
from PIL import Image
import yaml
import zmq, zmq.asyncio

from messages import *
import utils

utils.setup_exceptions()

MODULE_DIR = Path(__file__).parent.resolve()
STATIC_PATH = MODULE_DIR / 'static'
TEMPLATES_PATH = MODULE_DIR / 'templates'
WORKER_PATH = MODULE_DIR / 'worker.py'

logger = logging.getLogger('app')

ctx = zmq.asyncio.Context()
loop = zmq.asyncio.ZMQEventLoop()
asyncio.set_event_loop(loop)


@aiohttp_jinja2.template('index.html')
async def root(request):
    return {}


async def output_image(request):
    buf = io.BytesIO()
    utils.as_pil(request.app.input_arr).save(buf, format='png')
    headers = {'Cache-Control': 'no-cache'}
    return web.Response(content_type='image/png', body=buf.getvalue(), headers=headers)


async def upload(request):
    msg = await request.post()
    data = binascii.a2b_base64(msg['data'].partition(',')[2])
    image = Image.open(io.BytesIO(data)).convert('RGB')
    thumbnail_msg = None
    if msg['slot'] == 'input':
        current_image = np.uint8(image.resize(request.app.input_arr.shape[:2][::-1],
                                              Image.LANCZOS))
        request.app.input_arr = current_image
        out_msg = SetImages(input_image=current_image)
    elif msg['slot'] == 'style':
        current_image = np.uint8(utils.resize_to_fit(image, int(msg['size'])))
        request.app.style_size = msg['size']
        out_msg = SetImages(style_image=current_image)
        request.app.style_image = image
        make_thumbnails(request.app)
        thumbnail_msg = dict(type='thumbnails', style=request.app.style_image.thumbnail_url_)
    elif msg['slot'] == 'content':
        current_image = np.uint8(utils.resize_to_fit(image, int(msg['size'])))
        out_msg = SetImages(current_image.shape[:2], SetImages.RESAMPLE, current_image)
        request.app.its_per_s.clear()
        request.app.content_image = image
        make_thumbnails(request.app)
        send_websocket(request.app, dict(type='newSize', height=current_image.shape[0],
                                         width=current_image.shape[1]))
        request.app.params['size'] = max(current_image.shape[:2])
        send_websocket(request.app, dict(type='newParams', params=get_params(app)))
        thumbnail_msg = dict(type='thumbnails', content=request.app.content_image.thumbnail_url_)
    request.app.sock_out.send_pyobj(out_msg)
    if thumbnail_msg is not None:
        send_websocket(request.app, thumbnail_msg)
    return web.Response()


def make_thumbnails(app, size=300):
    header = 'data:image/jpeg;base64,'
    if not hasattr(app.content_image, 'thumbnail_'):
        small = utils.resize_to_fit(app.content_image, size, scale_up=False)
        buf = io.BytesIO()
        small.save(buf, format='jpeg', quality=85)
        app.content_image.thumbnail_url_ = header + binascii.b2a_base64(buf.getvalue()).decode()
    if not hasattr(app.style_image, 'thumbnail_'):
        small = utils.resize_to_fit(app.style_image, size, scale_up=False)
        buf = io.BytesIO()
        small.save(buf, format='jpeg', quality=85)
        app.style_image.thumbnail_url_ = header + binascii.b2a_base64(buf.getvalue()).decode()


async def websocket(request):
    app = request.app
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app.wss.append(ws)

    if app.worker_ready:
        send_websocket(app, dict(type='workerReady'))
    send_websocket(app, dict(type='newParams', params=get_params(app)))
    h, w = app.input_arr.shape[:2]
    send_websocket(app, dict(type='newSize', height=h, width=w))
    send_websocket(app, dict(type='state', running=app.running))
    make_thumbnails(app)
    send_websocket(app, dict(type='thumbnails',
                             content=app.content_image.thumbnail_url_,
                             style=app.style_image.thumbnail_url_))

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            msg = json.loads(msg.data)
            if 'type' not in msg:
                logger.error('Received an WebSocket message of unknown type.')
            if msg['type'] == 'applyParams':
                process_params(app, msg)
            elif msg['type'] == 'pause':
                app.sock_out.send_pyobj(PauseIteration())
                app.running = False
                send_websocket(app, dict(type='state', running=app.running))
            elif msg['type'] == 'reset':
                image = np.uint8(np.random.uniform(0, 255, app.input_arr.shape))
                app.input_arr = image
                app.sock_out.send_pyobj(SetImages(input_image=image, reset_state=True))
            elif msg['type'] == 'restartWorker':
                app.running = False
                send_websocket(app, dict(type='state', running=app.running))
                app.sock_out.send_pyobj(Shutdown())
            elif msg['type'] == 'start':
                app.sock_out.send_pyobj(StartIteration())
                app.running = True
                send_websocket(app, dict(type='state', running=app.running))
            else:
                logger.error('Received an WebSocket message of unknown type.')
        else:
            await ws.close()

    request.app.wss.remove(ws)
    return ws


def send_websocket(app, msg):
    for ws in app.wss:
        ws.send_json(msg)


def get_params(app):
    return yaml.dump(app.params)


def process_params(app, msg):
    try:
        params = yaml.safe_load(msg['params'])

        if params['size'] != max(app.input_arr.shape):
            new_size = utils.fit_into_square(app.input_arr.shape[:2], params['size'], True)
            content_image = app.content_image.resize(new_size[::-1], Image.LANCZOS)
            app.its_per_s.clear()

            if app.i == 0:
                input_image = np.uint8(np.random.uniform(0, 255, new_size + (3,)))
                app.input_arr = input_image
            else:
                input_image = utils.resample_hwc(app.input_arr, new_size)
                app.input_arr = input_image

            msg_out = SetImages(new_size, input_image, np.uint8(content_image))
            app.sock_out.send_pyobj(msg_out)
            send_websocket(app, dict(type='newSize', height=new_size[0], width=new_size[1]))

        app.sock_out.send_pyobj(SetOptimizer(params['optimizer'], params['optimizer_step_size']))

        weights = {}
        for loss_name in SetWeights.loss_names:
            weights[loss_name] = {}
            for layer, weight in params['weights'][0][loss_name].items():
                if layer not in app.layers:
                    raise ValueError('Invalid layer name. Valid layer names are: %s' % \
                                     ', '.join(app.layers))
                weights[loss_name][layer] = weight

        scalar_weights = {}
        for loss_name in SetWeights.scalar_loss_names:
            scalar_weights[loss_name] = params['weights'][1][loss_name]

        app.sock_out.send_pyobj(SetWeights(weights, scalar_weights))

        app.params = params
    except yaml.YAMLError:
        pass  # TODO: send an error back to the user
    except KeyError:
        pass  # TODO: send an error back to the user
    except ValueError:
        pass  # TODO: send an error back to the user
    finally:
        msg = dict(type='newParams', params=get_params(app))
        send_websocket(app, msg)


def init_params(app):
    app.content_image = Image.open(str(MODULE_DIR / app.config['initial_content'])).convert('RGB')
    app.style_image = Image.open(str(MODULE_DIR / app.config['initial_style'])).convert('RGB')
    size = app.config.getint('initial_size')

    app.params['size'] = size
    app.style_size = size
    app.params['optimizer'] = 'lbfgs'
    app.params['optimizer_step_size'] = SetOptimizer.step_sizes['lbfgs']
    with open(str(MODULE_DIR / app.config['initial_weights'])) as w:
        app.params['weights'] = yaml.load(w)


def init_arrays(app):
    content = utils.resize_to_fit(app.content_image, app.params['size'])
    style = utils.resize_to_fit(app.style_image, app.style_size)
    if not hasattr(app, 'input_arr'):
        w, h = content.size
        app.input_arr = np.uint8(np.random.uniform(0, 255, (h, w, 3)))

    if max(app.input_arr.shape[:2]) != app.params['size']:
        size = utils.fit_into_square(app.input_arr.shape[:2], app.params['size'])
        app.input_arr = utils.resample_hwc(app.input_arr, size)

    msg = SetImages(None, app.input_arr, np.uint8(content), np.uint8(style))
    app.sock_out.send_pyobj(msg)

    app.sock_out.send_pyobj(SetWeights(*app.params['weights']))


def process_iterate(app, recv_msg):
    # Update the average iterates per second value
    it_time = time.perf_counter()
    if recv_msg.i:
        app.its_per_s(1 / (it_time - app.last_it_time))
    else:
        app.its_per_s.clear()
    app.i = recv_msg.i
    app.last_it_time = it_time

    # Compute RMS difference of iterates
    step_size = np.nan
    if recv_msg.i > 0 and recv_msg.image.shape == app.input_arr.shape:
        diff = recv_msg.image - app.input_arr
        step_size = np.sqrt(np.mean(diff**2))

    logger.info('iterate %d received, loss: %g, step size: %g',
                recv_msg.i, recv_msg.trace['loss'], step_size)

    # Notify the client that an iterate was received
    msg = dict(type='iterateInfo', i=recv_msg.i, trace=recv_msg.trace,
               stepSize=float(step_size), itsPerS=app.its_per_s())
    send_websocket(app, msg)
    app.input_arr = recv_msg.image


async def process_messages(app):
    while True:
        recv_msg = await app.sock_in.recv_pyobj()
        if isinstance(recv_msg, Iterate):
            process_iterate(app, recv_msg)

        elif isinstance(recv_msg, Shutdown):
            raise KeyboardInterrupt()

        elif isinstance(recv_msg, WorkerReady):
            app.worker_ready = True
            app.layers = recv_msg.layers
            send_websocket(app, dict(type='workerReady'))

        elif isinstance(recv_msg, GetImages):
            init_arrays(app)

        else:
            logger.error('Unknown message type received over ZeroMQ.')


async def monitor_worker(app):
    while True:
        if app.worker_proc is None or app.worker_proc.poll() is not None:
            app.running = False
            app.worker_ready = False
            app.worker_proc = subprocess.Popen([str(WORKER_PATH)])
            send_websocket(app, dict(type='state', running=app.running))
            init_arrays(app)
        await asyncio.sleep(0.1)


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_out = ctx.socket(zmq.PUSH)
    app.sock_in.bind(app.config['app_socket'])
    app.sock_out.connect(app.config['worker_socket'])
    app.wss = []
    app.running = False
    app.last_it_time = 0
    app.its_per_s = utils.DecayingMean()
    app.params = {}
    app.layers = []
    init_params(app)
    init_arrays(app)
    app.i = 0
    app.worker_proc = None
    app.mw_future = asyncio.ensure_future(monitor_worker(app))
    app.pm_future = asyncio.ensure_future(process_messages(app))


async def cleanup_tasks(app):
    app.pm_future.cancel()
    app.mw_future.cancel()
    app.sock_out.send_pyobj(Shutdown())
    try:
        app.worker_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        app.worker_proc.terminate()


def init():
    app = web.Application()
    app.config = utils.read_config()

    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(TEMPLATES_PATH)))
    app.router.add_route('GET', '/', root)
    app.router.add_route('GET', '/output', output_image)
    app.router.add_route('POST', '/upload', upload)
    app.router.add_route('GET', '/websocket', websocket)
    app.router.add_static('/', STATIC_PATH)

    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    return app

app = init()


def main():
    """The main function."""
    debug = app.config.getboolean('debug', False)
    if debug:
        utils.setup_exceptions(mode='Context')
    utils.setup_logging(debug)

    try:
        web.run_app(app, host=app.config['http_host'], port=app.config['http_port'],
                    shutdown_timeout=1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
