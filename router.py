#!/usr/bin/env python3

"""A request router for hosting multiple instances of Style Transfer."""

# pylint: disable=redefined-outer-name

import asyncio
import os
from pathlib import Path
import pickle
import time

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
import zmq, zmq.asyncio

from error_pages import ErrorPages
from messages import *
import utils

utils.setup_exceptions()

MODULE_DIR = Path(__file__).parent.resolve()
STATE_FILE = 'router_state.pkl'
STATIC_PATH = MODULE_DIR / 'static'
TEMPLATES_PATH = MODULE_DIR / 'templates'

logger = logging.getLogger('router')

ctx = zmq.asyncio.Context()
loop = zmq.asyncio.ZMQEventLoop()
asyncio.set_event_loop(loop)


class AppInstance:
    def __init__(self, addr, socket, host, port, app_id, session_id=None):
        self.addr = addr
        self.socket = socket
        self.host = host
        self.port = port
        self.app_id = app_id
        self.session_id = session_id
        self.last_access = time.monotonic()
        self.last_ping = time.monotonic()


class RouterState:
    def __init__(self, addrs, sessions):
        self.addrs = addrs
        self.sessions = sessions


async def proxy(request):
    if 'session_id' in request.cookies and request.cookies['session_id'] in request.app.sessions:
        session_id = request.cookies['session_id']
        set_session_id = False
    else:
        session_id = os.urandom(8).hex()
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

    try:
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
    except aiohttp.errors.ClientError:
        logger.debug('Expiring session %s on %s', inst.session_id, inst.addr)
        inst.socket.send_pyobj(Reset())
        del request.app.sessions[session_id]
        raise web.HTTPInternalServerError()

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
    try:
        async with aiohttp.ws_connect(url) as ws_app:
            await ws_user.prepare(request)
            copy_coros = copy_ws(inst, ws_app, ws_user), copy_ws(inst, ws_user, ws_app)
            _, pending = await asyncio.wait(copy_coros, return_when=asyncio.FIRST_COMPLETED)
            [fut.cancel() for fut in pending]
            return ws_user
    except aiohttp.errors.ClientError:
        logger.debug('Expiring session %s on %s', inst.session_id, inst.addr)
        inst.socket.send_pyobj(Reset())
        del request.app.sessions[session_id]


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
                sess = None
                for session_id, inst_b in app.sessions.items():
                    if inst_a.app_id == inst_b.app_id:
                        sess = session_id
                        break
                if sess:
                    inst = app.sessions[sess]
                    logger.debug('Expiring session %s on %s', inst.session_id, inst.addr)
                    inst.socket.send_pyobj(Reset())
                    del app.sessions[sess]

        elif isinstance(msg, AppUp):
            if msg.addr not in app.addrs or app.addrs[msg.addr].app_id != msg.app_id:
                logger.info('%s', msg)
                if msg.addr in app.addrs:
                    inst = app.addrs[msg.addr]
                    if inst.session_id in app.sessions:
                        logger.debug('Expiring session %s on %s', inst.session_id, inst.addr)
                        del app.sessions[inst.session_id]
                    inst.socket.close()
                socket = ctx.socket(zmq.PUSH)
                socket.connect(msg.addr)
                socket.send_pyobj(Reset())
                inst = AppInstance(msg.addr, socket, msg.host, msg.port, msg.app_id)
                app.addrs[msg.addr] = inst
            else:
                app.addrs[msg.addr].last_ping = time.monotonic()

        else:
            logger.error('Unknown message type received over ZeroMQ.')


async def expire_state(app, run_once=False):
    timeout = app.config.getint('router_session_timeout')
    while True:
        now = time.monotonic()
        addr_to_del = None
        for addr, inst in app.addrs.items():
            if inst.session_id is not None and inst.last_access < now - timeout:
                logger.debug('Expiring session %s on %s', inst.session_id, addr)
                if inst.socket and not inst.socket.closed:
                    inst.socket.send_pyobj(Reset())
                if inst.session_id in app.sessions:
                    del app.sessions[inst.session_id]
                inst.session_id = None
            if inst.last_ping < now - timeout:
                addr_to_del = addr
        if addr_to_del:
            logger.debug('Ping timeout for instance %s', addr_to_del)
            del app.addrs[addr_to_del]
        if run_once:
            return
        await asyncio.sleep(1)


async def startup_tasks(app):
    app.sock_in = ctx.socket(zmq.PULL)
    app.sock_in.bind(app.config['router_socket'])
    app.addrs = {}
    app.sessions = {}
    try:
        state = pickle.load(open(STATE_FILE, 'rb'))
        app.addrs = state.addrs
        app.sessions = state.sessions
        await expire_state(app, run_once=True)
        for inst in state.addrs.values():
            inst.socket = ctx.socket(zmq.PUSH)
            inst.socket.connect(inst.addr)
    except EOFError as err:
        logger.warning('Unable to load state file: %s', err)
    except FileNotFoundError:
        pass
    app.expire_task = asyncio.Task(expire_state(app))
    app.pm_task = asyncio.Task(process_messages(app))


async def cleanup_tasks(app):
    app.pm_task.cancel()
    app.expire_task.cancel()


def init(args):
    app = web.Application(middlewares=[ErrorPages()])
    app.config = utils.read_config(args)
    app.debug_level = app.config.getint('debug', 0)
    if args.debug:
        app.debug_level += args.debug

    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(TEMPLATES_PATH)))
    app.router.add_route('GET', '/', proxy)
    app.router.add_route('GET', '/output', proxy)
    app.router.add_route('POST', '/upload', proxy)
    app.router.add_route('GET', '/websocket', proxy_ws)
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
        web.run_app(app, host=app.config['router_host'], port=int(app.config['router_port']),
                    shutdown_timeout=1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('Shutting down router.')
        ctx.destroy(0)
        for inst in app.addrs.values():
            inst.socket = None
        state = RouterState(app.addrs, app.sessions)
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
