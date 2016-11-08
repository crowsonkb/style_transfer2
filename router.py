#!/usr/bin/env python3

"""A request router for hosting multiple instances of Style Transfer."""

# pylint: disable=redefined-outer-name

import asyncio
import os
from pathlib import Path
import time

import aiohttp
from aiohttp import web
import zmq, zmq.asyncio

from messages import *
import utils

utils.setup_exceptions()

MODULE_DIR = Path(__file__).parent.resolve()

logger = logging.getLogger('router')

ctx = zmq.asyncio.Context()
loop = zmq.asyncio.ZMQEventLoop()
asyncio.set_event_loop(loop)


class AppInstance:
    def __init__(self, addr, socket, host, port, session_id):
        self.addr = addr
        self.socket = socket
        self.host = host
        self.port = port
        self.session_id = session_id
        self.last_access = time.monotonic()


async def proxy(request):
    if 'session_id' in request.cookies and request.cookies['session_id'] in request.app.sessions:
        session_id = request.cookies['session_id']
        set_session_id = False
    else:
        session_id = os.urandom(16).hex()
        inst = None
        for instance in request.app.addrs.values():
            if instance.session_id is None:
                inst = instance
                logger.debug('New session: %s on %s', session_id, inst.addr)
                inst.session_id = session_id
                request.app.sessions[session_id] = inst
                break
        if inst is None:
            raise web.HTTPServiceUnavailable()
        set_session_id = True

    inst = request.app.sessions[session_id]
    inst.last_access = time.monotonic()

    async with aiohttp.ClientSession() as sess:
        url = 'http://%s:%d%s' % (inst.host, inst.port, request.rel_url)
        if request.method == 'GET':
            async with sess.get(url, headers=request.headers) as resp:
                data = await resp.read()
                resp = web.Response(body=data, headers=resp.headers, status=resp.status)
        elif request.method == 'POST':
            data = await request.read()
            async with sess.post(url, headers=request.headers, data=data) as resp:
                data = await resp.read()
                resp = web.Response(body=data, headers=resp.headers, status=resp.status)
        else:
            raise web.HTTPForbidden()

    if set_session_id:
        resp.set_cookie('session_id', session_id)
    return resp


async def proxy_ws(request):
    if 'session_id' not in request.cookies:
        raise aiohttp.web.HTTPForbidden()
    if request.cookies['session_id'] not in request.app.sessions:
        raise aiohttp.web.HTTPForbidden()
    session_id = request.cookies['session_id']
    inst = request.app.sessions[session_id]
    url = 'http://%s:%d%s' % (inst.host, inst.port, '/websocket')
    ws_user = web.WebSocketResponse()
    async with aiohttp.ws_connect(url) as ws_app:
        await ws_user.prepare(request)
        copy_coros = copy_ws(inst, ws_app, ws_user), copy_ws(inst, ws_user, ws_app)
        _, pending = await asyncio.wait(copy_coros, return_when=asyncio.FIRST_COMPLETED)
        [fut.cancel() for fut in pending]
        return ws_user


async def copy_ws(inst, a, b):
    async for msg in b:
        try:
            if msg.type == aiohttp.WSMsgType.TEXT:
                a.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                a.send_bytes(msg.data)
            inst.last_access = time.monotonic()
        except RuntimeError:
            break
    await a.close()


async def process_messages(app):
    while True:
        msg = await app.sock_in.recv_pyobj()

        if isinstance(msg, AppDown):
            logger.info('%s', msg)
            if msg.addr in app.addrs:
                inst_a = app.addrs[msg.addr]
                inst_a.socket.close()
                del app.addrs[msg.addr]
                for session_id, inst_b in app.sessions.items():
                    if inst_a == inst_b:
                        del app.sessions[session_id]

        elif isinstance(msg, AppUp):
            if msg.addr not in app.addrs:
                logger.info('%s', msg)
                socket = ctx.socket(zmq.PUSH)
                socket.connect(msg.addr)
                inst = AppInstance(msg.addr, socket, msg.host, msg.port, None)
                app.addrs[msg.addr] = inst

        else:
            logger.error('Unknown message type received over ZeroMQ.')


async def expire_sessions(app):
    while True:
        now = time.monotonic()
        for addr, inst in app.addrs.items():
            if inst.session_id is not None and inst.last_access < now - 1:
                logger.debug('Expiring session %s on %s', inst.session_id, addr)
                inst.socket.send_pyobj(Reset())
                del app.sessions[inst.session_id]
                inst.session_id = None
        await asyncio.sleep(1)


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_in.bind(app.config['router_socket'])
    app.addrs = {}
    app.sessions = {}
    app.expire_future = asyncio.ensure_future(expire_sessions(app))
    app.pm_future = asyncio.ensure_future(process_messages(app))


async def cleanup_tasks(app):
    app.pm_future.cancel()
    app.expire_future.cancel()


def init():
    app = web.Application()
    app.config = utils.read_config()
    app.router.add_route('GET', '/websocket', proxy_ws)
    app.router.add_route('GET', r'/{a:.*}', proxy)
    app.router.add_route('POST', r'/{a:.*}', proxy)
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
        web.run_app(app, host=app.config['router_host'], port=int(app.config['router_port']),
                    shutdown_timeout=1)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
