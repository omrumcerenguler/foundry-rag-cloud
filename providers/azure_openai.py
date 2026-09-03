"""Small dependency-free Azure OpenAI provider adapters."""

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from core.ports import BaseChatProvider, BaseEmbeddingProvider

urlopen = getattr(urllib.request, "urlopen")
logger = logging.getLogger(__name__)


class AzureProviderError(RuntimeError):
    """Base error for Azure provider failures."""


class AzureAuthenticationError(AzureProviderError):
    """The Azure credential was rejected."""


AzureAuthError = AzureAuthenticationError


class AzureAuthorizationError(AzureProviderError):
    """The Azure credential lacks permission for the requested resource."""


class AzureRateLimitError(AzureProviderError):
    """Azure rejected a request because of rate limiting."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AzureServerError(AzureProviderError):
    """Azure returned a transient server-side error."""


def _retry_after(value: str | None) -> float | None:
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return min(10.0, max(0.0, float(value)))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return min(
                10.0, max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            )
        except (TypeError, ValueError, OverflowError):
            return None


class _AzureProvider:
    """Shared authenticated JSON request implementation."""

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize a resource endpoint copied from Azure Foundry."""
        normalized = endpoint.strip().rstrip("/")
        if normalized.lower().endswith("/openai/v1"):
            normalized = normalized[: -len("/openai/v1")]
        return normalized.rstrip("/")

    def __init__(self, endpoint: str, api_key: str, api_version: str) -> None:
        if not endpoint or not api_key:
            raise ValueError("Azure endpoint and API key are required")
        parsed_endpoint = urlsplit(endpoint)
        hostname = parsed_endpoint.hostname or ""
        local_hosts = {"localhost", "127.0.0.1", "::1", "example"}
        if parsed_endpoint.scheme == "https":
            pass
        elif parsed_endpoint.scheme == "http" and (
            hostname in local_hosts
            or hostname.endswith(".localhost")
            or hostname.endswith(".test")
        ):
            pass
        else:
            raise ValueError(
                "Azure endpoint must use https, or http for local/mock endpoints"
            )
        if not hostname:
            raise ValueError("Azure endpoint must include a hostname")
        self.endpoint = self._normalize_endpoint(endpoint)
        self.api_key = api_key
        self.api_version = api_version

    def _post(
        self, resource: str, deployment: str, payload: dict[str, object]
    ) -> dict[str, object]:
        url = f"{self.endpoint}/openai/deployments/{deployment}/{resource}"
        for attempt in range(4):
            request = Request(
                f"{url}?api-version={self.api_version}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "api-key": self.api_key},
                method="POST",
            )
            try:
                open_url = urlopen
                with open_url(request, timeout=60) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise AzureProviderError("Azure response must be a JSON object")
                return result
            except HTTPError as exc:
                response_body = self._safe_error_body(exc)
                logger.warning(
                    "Azure request rejected: status=%s endpoint=%s body=%s",
                    exc.code,
                    self.endpoint,
                    response_body,
                )
                retry_after = _retry_after(
                    exc.headers.get("Retry-After") if exc.headers is not None else None
                )
                if exc.code == 401:
                    raise AzureAuthenticationError(
                        "Azure authentication failed"
                    ) from exc
                if exc.code == 403:
                    raise AzureAuthorizationError("Azure authorization failed") from exc
                if exc.code == 429:
                    if attempt < 3:
                        delay = retry_after if retry_after is not None else 2.0**attempt
                        time.sleep(delay)
                        continue
                    raise AzureRateLimitError(
                        "Azure rate limit exceeded", retry_after
                    ) from exc
                if 500 <= exc.code <= 599:
                    if attempt < 2:
                        time.sleep(2.0**attempt)
                        continue
                    raise AzureServerError(f"Azure server error ({exc.code})") from exc
                raise AzureProviderError(f"Azure request failed ({exc.code})") from exc
            except (
                URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                raise AzureProviderError(f"Azure OpenAI request failed: {exc}") from exc
            except AzureProviderError:
                raise
            except Exception as exc:
                raise AzureProviderError(f"Azure OpenAI request failed: {exc}") from exc
        raise AzureProviderError("Azure request failed after retries")

    def _safe_error_body(self, error: HTTPError) -> str:
        """Read a bounded error body while ensuring the configured key is absent."""
        try:
            body = error.read(4096).decode("utf-8", errors="replace")
        except (OSError, UnicodeError):
            body = str(error)
        return body.replace(self.api_key, "<redacted-api-key>")


class AzureOpenAIEmbeddingProvider(_AzureProvider, BaseEmbeddingProvider):
    """Create embeddings using an Azure OpenAI embedding deployment."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        dimension: int,
    ) -> None:
        super().__init__(endpoint, api_key, api_version)
        self.deployment = deployment
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the Azure deployment name."""
        return self.deployment

    def embed_text(self, text: str) -> list[float]:
        """Create an embedding for one text."""
        if not text.strip():
            raise ValueError("text must not be empty")
        response = self._post("embeddings", self.deployment, {"input": text})
        try:
            embedding = [float(value) for value in response["data"][0]["embedding"]]  # type: ignore[index]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Azure embedding response has an unsupported shape"
            ) from exc
        if len(embedding) != self._dimension:
            raise RuntimeError(
                f"Expected {self._dimension} dimensions, received {len(embedding)}"
            )
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings in input order."""
        return [self.embed_text(text) for text in texts]


class AzureOpenAIChatProvider(_AzureProvider, BaseChatProvider):
    """Create chat completions using an Azure OpenAI deployment."""

    def __init__(
        self, endpoint: str, api_key: str, api_version: str, deployment: str
    ) -> None:
        super().__init__(endpoint, api_key, api_version)
        self.deployment = deployment

    @property
    def model_name(self) -> str:
        """Return the Azure chat deployment name."""
        return self.deployment

    def generate_response(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Create one grounded chat completion."""
        response = self._post(
            "chat/completions",
            self.deployment,
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Azure chat response has an unsupported shape") from exc
        if not content:
            raise RuntimeError("Azure chat response was empty")
        return str(content)
