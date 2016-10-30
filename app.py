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
    image = np.float32(utils.resize_to_fit(image, int(msg['size'])))
    app.sock_out.send_pyobj(SetImage(msg['slot'], image))
    if msg['slot'] == 'content':
        new_input = np.random.uniform(0, 255, image.shape).astype(np.float32)
    else:
        new_input = np.random.uniform(0, 255, app.input_arr.shape).astype(np.float32)
    app.sock_out.send_pyobj(SetImage('input', new_input))
    return web.Response()


async def websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app.wss.append(ws)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            msg = json.loads(msg.data)
        else:
            await ws.close()

    request.app.wss.remove(ws)
    return ws


async def worker_test(app):
    app.input_arr = np.random.uniform(0, 255, (96, 96, 3)).astype(np.float32)
    app.sock_out.send_pyobj(SetImage('input', app.input_arr))
    msg = SetImage('content', np.float32(Image.open('../style_transfer/golden_gate.jpg').resize((96, 96), Image.LANCZOS)))
    app.sock_out.send_pyobj(msg)
    msg = SetImage('style', np.float32(Image.open('../style_transfer/seated-nude.jpg').resize((96, 96), Image.LANCZOS)))
    app.sock_out.send_pyobj(msg)
    app.sock_out.send_pyobj(SetWeights(dict(content=dict(conv4_2=1/15), style={'conv1_1':1, 'conv2_1':1, 'conv3_1':1, 'conv4_1':1}), {'tv': 10}))
    app.sock_out.send_pyobj(StartIteration())

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
            app.last_it_time = it_time

            # Compute RMS difference of iterates
            step_size = np.nan
            if recv_msg.i > 0 and recv_msg.image.shape == app.input_arr.shape:
                step_size = np.sqrt(np.mean(np.square(recv_msg.image - app.input_arr)))

            logger.info('iterate %d received, loss: %g, step size: %g',
                        recv_msg.i, recv_msg.loss, step_size)

            # Notify the client that an iterate was received
            for ws in app.wss:
                ws.send_json(dict(type='iterateInfo', i=recv_msg.i, loss=float(recv_msg.loss),
                                  stepSize=float(step_size), itsPerS=true_its_per_s))
            app.input_arr = recv_msg.image


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_out = ctx.socket(zmq.PUSH)
    app.sock_in.bind(app.config['app_socket'])
    app.sock_out.connect(app.config['worker_socket'])
    app.wss = []
    app.last_it_time = 0
    app.its_per_s = 0
    asyncio.ensure_future(worker_test(app))
    app.worker_proc = subprocess.Popen([str(WORKER_PATH)])


async def cleanup_tasks(app):
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
    web.run_app(app, host=app.config['http_host'], port=app.config['http_port'])

if __name__ == '__main__':
    main()
