from aiohttp import web
import aiohttp_jinja2


MESSAGES = {
    503: '''Style Transfer is temporarily unavailable due to high load. If you refresh the page
    or come back in about a minute, it might be available.''',
}


class ErrorPages:
    async def __call__(self, app, handler):
        async def middleware_handler(request):
            try:
                response = await handler(request)
            except web.HTTPException as err:
                response = err
            if response.status >= 400:
                context = dict(status_code=response.status,
                            reason=response.reason,
                            message=MESSAGES.get(response.status, ''))

                @aiohttp_jinja2.template('error.html', status=response.status)
                async def template_fn(request):
                    return context
                return await template_fn(request)
            return response
        return middleware_handler
