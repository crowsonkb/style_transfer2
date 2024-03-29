#!/usr/bin/env python3

"""The main module for a web application implementing neural style transfer using Caffe. See
"A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576)."""

# pylint: disable=redefined-outer-name

import asyncio
import binascii
import io
import logging
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
import numpy as np
from PIL import Image
import yaml
import zmq, zmq.asyncio

from error_pages import ErrorPages
from messages import *
import utils

utils.setup_exceptions()
utils.setup_signals()

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
    return {'max_size': request.app.config.getint('max_size', 9999),
            'ga_tracking_code': request.app.config.get('ga_tracking_code', ''),
            'top': open(request.app.config.get('top', '')).read()}


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
        input_image = SetImages.RESAMPLE
        if request.app.i <= 1:
            input_image = np.uint8(np.random.uniform(0, 255, current_image.shape[:2] + (3,)))
            request.app.input_arr = input_image
        out_msg = SetImages(current_image.shape[:2], input_image, current_image)
        request.app.its_per_s.clear()
        request.app.content_image = image
        make_thumbnails(request.app)
        send_websocket(request.app, dict(type='newSize', height=current_image.shape[0],
                                         width=current_image.shape[1]))
        request.app.params['size'] = max(current_image.shape[:2])
        send_websocket(request.app, dict(type='newParams', params=get_params(request.app)))
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
                app.input_was_reset = True
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
        try:
            ws.send_json(msg)
        except TypeError as err:
            logger.debug('TypeError: %s', err)
        except RuntimeError:
            pass


def get_params(app):
    return yaml.dump(app.params)


def process_params(app, msg):
    error_string = ''

    try:
        params = yaml.safe_load(msg['params'])

        max_size = app.config.getint('max_size', 9999)
        if params['size'] > max_size:
            raise ValueError('Size is over %d' % max_size)

        if params['size'] != max(app.input_arr.shape):
            new_size = utils.fit_into_square(app.input_arr.shape[:2], params['size'], True)
            content_image = app.content_image.resize(new_size[::-1], Image.LANCZOS)
            app.its_per_s.clear()

            if app.i <= 1:
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
                                     ', '.join(app.layers) + '.')
                weights[loss_name][layer] = float(weight)

        scalar_weights = {}
        for loss_name in SetWeights.scalar_loss_names:
            scalar_weights[loss_name] = float(params['weights'][1][loss_name])

        app.sock_out.send_pyobj(SetWeights(weights, scalar_weights))

        app.params = params
    except KeyError as err:
        error_string = err.__class__.__name__ + ': ' + str(err) + \
                       ': All required parameters were not found. Please don\'t delete parameters.'
    except Exception as err:
        error_string = err.__class__.__name__ + ': ' + str(err)
    finally:
        msg = dict(type='newParams', params=get_params(app), errorString=error_string)
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

    reset_state = False
    if app.input_arr is None:
        w, h = content.size
        app.input_arr = np.uint8(np.random.uniform(0, 255, (h, w, 3)))
        app.input_was_reset = True
        reset_state = True
    elif max(app.input_arr.shape[:2]) != app.params['size']:
        size = utils.fit_into_square(app.input_arr.shape[:2], app.params['size'])
        app.input_arr = utils.resample_hwc(app.input_arr, size)
        app.input_was_reset = False

    msg = SetImages(None, app.input_arr, np.uint8(content), np.uint8(style), reset_state)
    app.sock_out.send_pyobj(msg)

    app.sock_out.send_pyobj(SetWeights(*app.params['weights']))


def process_iterate(app, recv_msg):
    # Update the average iterates per second value
    it_time = time.perf_counter()
    if recv_msg.i == 1:
        app.its_per_s.clear()
    else:
        app.its_per_s(1 / (it_time - app.last_it_time))
    app.i = recv_msg.i
    app.last_it_time = it_time

    # Compute RMS difference of iterates
    step_size = 0
    if recv_msg.i > 1 and recv_msg.image.shape == app.input_arr.shape:
        diff = recv_msg.image - app.input_arr
        step_size = np.sqrt(np.mean(diff**2))

    logger.info('iterate %d received, loss: %g, step size: %g',
                recv_msg.i, recv_msg.trace['loss'], step_size)

    # Notify the client that an iterate was received
    if app.running and (not app.input_was_reset or recv_msg.i == 1):
        app.input_was_reset = False
        app.input_arr = recv_msg.image
        msg = dict(type='iterateInfo', i=recv_msg.i, trace=recv_msg.trace,
                   stepSize=float(step_size), itsPerS=app.its_per_s())
        send_websocket(app, msg)


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
            if app.sock_router:
                app.sock_router.send_pyobj(AppUp(app.config['app_socket'],
                                                 app.config['http_host'],
                                                 int(app.config['http_port']),
                                                 app.id))

        elif isinstance(recv_msg, GetImages):
            init_arrays(app)

        elif isinstance(recv_msg, Reset):
            app.sock_out.send_pyobj(PauseIteration())
            app.running = False
            init_params(app)
            app.input_arr = None
            init_arrays(app)

        else:
            logger.error('Unknown message type received over ZeroMQ.')


async def ping_router(app):
    while True:
        if app.worker_ready:
            app.sock_router.send_pyobj(AppUp(app.config['app_socket'],
                                             app.config['http_host'],
                                             int(app.config['http_port']),
                                             app.id))
        await asyncio.sleep(5)


async def monitor_worker(app):
    while True:
        if app.worker_proc is None or app.worker_proc.poll() is not None:
            app.running = False
            app.worker_ready = False
            app.worker_proc = subprocess.Popen([str(WORKER_PATH)] + sys.argv[1:])
            send_websocket(app, dict(type='state', running=app.running))
            init_arrays(app)
        await asyncio.sleep(0.1)


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_out = ctx.socket(zmq.PUSH)
    app.sock_router = None
    app.sock_in.bind(app.config['app_socket'])
    app.sock_out.connect(app.config['worker_socket'])
    if 'router_socket' in app.config:
        app.sock_router = ctx.socket(zmq.PUSH)
        app.sock_router.connect(app.config['router_socket'])
    app.id = os.urandom(8).hex()
    app.wss = []
    app.running = False
    app.last_it_time = 0
    app.its_per_s = utils.DecayingMean()
    app.params = {}
    app.layers = []
    app.input_arr = None
    init_params(app)
    init_arrays(app)
    app.i = 0
    app.worker_proc = None
    app.mw_task = asyncio.Task(monitor_worker(app))
    app.pm_task = asyncio.Task(process_messages(app))
    if app.sock_router:
        app.pr_task = asyncio.Task(ping_router(app))


async def cleanup_tasks(app):
    if app.sock_router:
        app.pr_task.cancel()
        app.sock_router.send_pyobj(AppDown(app.config['app_socket'], app.id))
    app.pm_task.cancel()
    app.mw_task.cancel()
    app.sock_out.send_pyobj(Shutdown())
    try:
        app.worker_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        app.worker_proc.terminate()
        app.worker_proc.wait()


def init(args):
    config = utils.read_config(args)
    template_vars = {'ga_tracking_code': config.get('ga_tracking_code', '')}
    app = web.Application(middlewares=[ErrorPages(template_vars)])
    app.config = config
    app.debug_level = app.config.getint('debug', 0)
    if args.debug:
        app.debug_level += args.debug

    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(TEMPLATES_PATH)))
    app.router.add_route('GET', '/', root)
    app.router.add_route('GET', '/output', output_image)
    app.router.add_route('POST', '/upload', upload)
    app.router.add_route('GET', '/websocket', websocket)
    app.router.add_static('/', STATIC_PATH)

    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    return app


def main():
    """The main function."""
    args = utils.parse_args(__doc__)
    app = init(args)
    if app.debug_level:
        utils.setup_exceptions(mode='Context')
        app['debug'] = True
    if app.debug_level >= 2:
        loop.set_debug(True)
    utils.setup_logging(app.debug_level)

    try:
        web.run_app(app, host=app.config['http_host'], port=int(app.config['http_port']),
                    shutdown_timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('Shutting down app.')
        ctx.destroy(0)

if __name__ == '__main__':
    main()
