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

logger = logging.getLogger(__name__)

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
    current_image = np.float32(utils.resize_to_fit(image, int(msg['size'])))
    if msg['slot'] == 'style':
        out_msg = SetImages(style_image=current_image)
        request.app.style_image = image
    elif msg['slot'] == 'content':
        out_msg = SetImages(current_image.shape[:2], SetImages.RESAMPLE, current_image)
        request.app.content_image = image
    request.app.sock_out.send_pyobj(out_msg)
    return web.Response()


async def websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app.wss.append(ws)

    send_websocket(app, dict(type='newParams', params=get_params(app)))
    send_websocket(app, dict(type='newSize', size=max(request.app.input_arr.shape)))

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            msg = json.loads(msg.data)
            if 'type' not in msg:
                logger.error('Received an WebSocket message of unknown type.')
            if msg['type'] == 'applyParams':
                process_params(request.app, msg)
            elif msg['type'] == 'pause':
                request.app.sock_out.send_pyobj(PauseIteration())
            elif msg['type'] == 'reset':
                image = np.float32(np.random.uniform(0, 255, request.app.input_arr.shape))
                request.app.input_arr = image
                request.app.sock_out.send_pyobj(SetImages(input_image=image, reset_state=True))
            elif msg['type'] == 'start':
                request.app.sock_out.send_pyobj(StartIteration())
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
            input_image = SetImages.RESAMPLE
            if app.i == 0:
                input_image = np.float32(np.random.uniform(0, 255, new_size + (3,)))
                app.input_arr = input_image
            msg_out = SetImages(new_size, input_image, np.float32(content_image))
            app.sock_out.send_pyobj(msg_out)
            send_websocket(app, dict(type='newSize', size=params['size']))
        app.weights = params['weights']
        app.sock_out.send_pyobj(SetWeights(*app.weights))
        app.params = params
    finally:
        msg = dict(type='newParams', params=get_params(app))
        send_websocket(app, msg)


def init_arrays(app):
    app.content_image = Image.open(str(MODULE_DIR / app.config['initial_content'])).convert('RGB')
    app.style_image = Image.open(str(MODULE_DIR / app.config['initial_style'])).convert('RGB')
    size = app.config.getint('initial_size')
    app.params['size'] = size

    content = utils.resize_to_fit(app.content_image, size)
    style = utils.resize_to_fit(app.style_image, size)
    w, h = content.size

    app.input_arr = np.float32(np.random.uniform(0, 255, (h, w, 3)))
    app.i = 0
    msg = SetImages(None, app.input_arr, np.float32(content), np.float32(style))
    app.sock_out.send_pyobj(msg)

    with open(str(MODULE_DIR / app.config['initial_weights'])) as w:
        app.params['weights'] = yaml.load(w)
    app.sock_out.send_pyobj(SetWeights(*app.params['weights']))


async def process_messages(app):
    while True:
        recv_msg = await app.sock_in.recv_pyobj()
        if isinstance(recv_msg, Iterate):
            # Update the average iterates per second value
            it_time = time.perf_counter()
            if recv_msg.i:
                app.its_per_s = 0.9 * app.its_per_s + 0.1 / (it_time - app.last_it_time)
                true_its_per_s = app.its_per_s / (1 - 0.9**recv_msg.i)
            else:
                app.its_per_s = 0
                true_its_per_s = 0
            app.i = recv_msg.i
            app.last_it_time = it_time

            # Compute RMS difference of iterates
            step_size = np.nan
            if recv_msg.i > 0 and recv_msg.image.shape == app.input_arr.shape:
                diff = recv_msg.image - app.input_arr
                step_size = np.sqrt(np.mean(diff**2))

            logger.info('iterate %d received, loss: %g, step size: %g',
                        recv_msg.i, recv_msg.loss, step_size)

            # Notify the client that an iterate was received
            snr = 10 * np.log10(recv_msg.image.size / recv_msg.loss)
            msg = dict(type='iterateInfo', i=recv_msg.i, loss=float(snr),
                       stepSize=float(step_size), itsPerS=true_its_per_s)
            send_websocket(app, msg)
            app.input_arr = recv_msg.image

        elif isinstance(recv_msg, Shutdown):
            raise KeyboardInterrupt()

        else:
            logger.error('Unknown message type received over ZeroMQ.')


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_out = ctx.socket(zmq.PUSH)
    app.sock_in.bind(app.config['app_socket'])
    app.sock_out.connect(app.config['worker_socket'])
    app.wss = []
    app.last_it_time = 0
    app.its_per_s = 0
    app.params = {}
    init_arrays(app)
    app.pm_future = asyncio.ensure_future(process_messages(app))
    app.worker_proc = subprocess.Popen([str(WORKER_PATH)])


async def cleanup_tasks(app):
    app.pm_future.cancel()
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
    app.router.add_route('GET', '/output.png', output_image)
    app.router.add_route('POST', '/upload', upload)
    app.router.add_route('GET', '/websocket', websocket)
    app.router.add_static('/', STATIC_PATH)

    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    return app

app = init()


def main():
    """The main function."""
    utils.setup_logging()
    try:
        web.run_app(app, host=app.config['http_host'], port=app.config['http_port'],
                    shutdown_timeout=1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
