"""Container health probe for the FastAPI process."""

import urllib.error
import urllib.request
from typing import cast
from urllib.parse import urlsplit

HEALTH_URL = "http://127.0.0.1:8000/health"


def _health_status(url: str) -> int:
    """Fetch a health URL only when its scheme is explicitly supported."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("healthcheck URL must use http or https")
    if not parsed.hostname:
        raise ValueError("healthcheck URL must include a hostname")
    request = urllib.request.Request(url, method="GET")
    open_url = getattr(urllib.request, "urlopen")
    with open_url(request, timeout=3) as response:
        return cast(int, response.status)


try:
    status = _health_status(HEALTH_URL)
except (
    urllib.error.HTTPError,
    urllib.error.URLError,
    TimeoutError,
    ValueError,
) as error:
    if isinstance(error, urllib.error.HTTPError):
        status = error.code
    else:
        status = 0

raise SystemExit(0 if status == 200 else 1)
