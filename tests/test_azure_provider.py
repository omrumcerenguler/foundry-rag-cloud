import io
import json
from email.message import Message
from urllib.error import HTTPError

import providers.azure_openai as azure
from providers.azure_openai import (
    AzureOpenAIEmbeddingProvider,
    AzureRateLimitError,
    _AzureProvider,
)


def test_foundry_v1_endpoint_is_normalized() -> None:
    provider = _AzureProvider(
        " https://resource.openai.azure.com/openai/v1/ ", "key", "v"
    )
    assert provider.endpoint == "https://resource.openai.azure.com"


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_429_retries_and_honors_retry_after(monkeypatch) -> None:
    headers = Message()
    headers["Retry-After"] = "0"
    failures = [HTTPError("url", 429, "rate", headers, io.BytesIO()) for _ in range(2)]
    calls: list[float] = []

    def fake_urlopen(_request, **_kwargs: object):
        if failures:
            raise failures.pop(0)
        return Response(json.dumps({"data": [{"embedding": [1.0, 0.0]}]}).encode())

    monkeypatch.setattr(azure, "urlopen", fake_urlopen)
    monkeypatch.setattr(azure.time, "sleep", calls.append)
    provider = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "embed", 2)
    assert provider.embed_text("hello") == [1.0, 0.0]
    assert calls == [0.0, 0.0]


def test_429_exhaustion_is_structured(monkeypatch) -> None:
    def fake_urlopen(_request, **_kwargs: object):
        headers = Message()
        headers["Retry-After"] = "0"
        raise HTTPError("url", 429, "rate", headers, io.BytesIO())

    monkeypatch.setattr(azure, "urlopen", fake_urlopen)
    monkeypatch.setattr(azure.time, "sleep", lambda _seconds: None)
    provider = AzureOpenAIEmbeddingProvider("https://example", "key", "v", "embed", 2)
    try:
        provider.embed_text("hello")
    except AzureRateLimitError as exc:
        assert exc.retry_after == 0.0
    else:
        raise AssertionError("expected AzureRateLimitError")
