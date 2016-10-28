#!/usr/bin/env python3

"""An under-development web application."""

import logging
from pathlib import Path

from aiohttp import web
import aiohttp_jinja2
import jinja2

MODULE_PATH = Path(__file__).parent
STATIC_PATH = MODULE_PATH / 'static'


@aiohttp_jinja2.template('index.html')
async def root(request):
    return {}


def init():
    app = web.Application()  # pylint: disable=redefined-outer-name
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(MODULE_PATH / 'templates')))
    app.router.add_route('GET', '/', root)
    app.router.add_static('/', STATIC_PATH)
    return app

app = init()


def main():
    """The main function."""
    logging.basicConfig(level=logging.DEBUG)
    web.run_app(app)

if __name__ == '__main__':
    main()
