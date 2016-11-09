"""aiohttp middleware to serve HTTP error pages as jinja2 templates."""

from aiohttp import web
import aiohttp_jinja2


MESSAGES = {
    503: '''Style Transfer is temporarily unavailable due to high load. If you refresh the page
    or come back in about a minute, it might be available.''',
}

TEMPLATE = 'error.html'


class ErrorPages:
    """aiohttp middleware to serve HTTP error pages as jinja2 templates."""
    def __init__(self, template_vars=None):
        self.template_vars = template_vars
        if template_vars is None:
            self.template_vars = {}

    async def __call__(self, app, handler):
        """Middleware factory method."""

        async def error_handler(request):
            """Handler to serve HTTP error pages as jinja2 templates."""

            try:
                response = await handler(request)
            except web.HTTPException as err:
                response = err
            if response.status >= 400:
                context = dict(status_code=response.status,
                               reason=response.reason,
                               message=MESSAGES.get(response.status, ''))
                context.update(self.template_vars)

                @aiohttp_jinja2.template(TEMPLATE, status=response.status)
                async def template_fn(request):
                    return context
                return await template_fn(request)
            return response
        return error_handler
