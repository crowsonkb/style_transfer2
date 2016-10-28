#!/usr/bin/env python3

"""An under-development web application."""

# pylint: disable=redefined-outer-name

import asyncio
import configparser
import logging
from pathlib import Path
import subprocess

from aiohttp import web
import aiohttp_jinja2
import jinja2
import numpy as np
import zmq, zmq.asyncio

import messages
import utils

MODULE_DIR = Path(__file__).parent.resolve()
STATIC_PATH = MODULE_DIR / 'static'
WORKER_PATH = MODULE_DIR / 'worker.py'

logger = logging.getLogger(__name__)

ctx = zmq.asyncio.Context()
loop = zmq.asyncio.ZMQEventLoop()
asyncio.set_event_loop(loop)


@aiohttp_jinja2.template('index.html')
async def root(request):
    return {}


async def worker_test(app):
    msg = messages.SetImage('input', np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32))
    app.sock_out.send_pyobj(msg)
    msg = messages.SetImage('content', np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32))
    app.sock_out.send_pyobj(msg)
    msg = messages.SetImage('style', np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32))
    app.sock_out.send_pyobj(msg)
    app.sock_out.send_pyobj(messages.StartIteration())

    while True:
        msg = await app.sock_in.recv_pyobj()
        logger.info('iterate %d, received, loss: %g', msg.i, msg.loss)
        utils.as_pil(msg.image).save('debug.png')


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_out = ctx.socket(zmq.PUSH)
    app.sock_in.bind(app.config['app_socket'])
    app.sock_out.connect(app.config['worker_socket'])
    asyncio.ensure_future(worker_test(app))
    app.worker_proc = subprocess.Popen([str(WORKER_PATH)])


async def cleanup_tasks(app):
    app.worker_proc.terminate()


def init():
    app = web.Application()
    cp = configparser.ConfigParser()
    cp.read(str(MODULE_DIR / 'config.ini'))
    app.config = cp['DEFAULT']

    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(MODULE_DIR / 'templates')))
    app.router.add_route('GET', '/', root)
    app.router.add_static('/', STATIC_PATH)

    app.on_startup.append(startup_tasks)
    app.on_cleanup.append(cleanup_tasks)
    return app

app = init()


def main():
    """The main function."""
    logging.basicConfig(level=logging.DEBUG, format=utils.logging_format)
    logging.captureWarnings(True)
    web.run_app(app)

if __name__ == '__main__':
    main()
