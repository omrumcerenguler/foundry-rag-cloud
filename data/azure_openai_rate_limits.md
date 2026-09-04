# Azure OpenAI Rate Limits and Retry Strategy

## Overview

Azure OpenAI deployments enforce two primary quota dimensions: Tokens Per Minute (TPM) and Requests Per Minute (RPM). Both limits are configured per deployment in the Azure resource and apply independently, meaning a request can be rejected for exceeding either dimension even if the other has capacity remaining. Production systems must treat both as hard ceilings rather than soft guidelines, since Azure enforces them at the gateway level before the request reaches the underlying model.

## TPM and RPM Quotas

TPM quotas count both prompt tokens and completion tokens against the same rolling window, typically measured over a sixty-second interval. RPM quotas count the number of HTTP requests regardless of payload size. A deployment provisioned for 10,000 TPM can be exhausted by a small number of large-context requests just as easily as a large number of small requests. Capacity planning should model average tokens per request, expected concurrency, and burst patterns rather than relying on a single average throughput number.

## The 429 Status Code

When either quota is exceeded, Azure OpenAI returns an HTTP 429 Too Many Requests response. This status code is distinct from authentication failures (401), authorization failures (403), or server-side failures (5xx), and each category demands a different recovery strategy. Retrying a 401 or 403 without fixing credentials wastes attempts and delays failure visibility. Retrying a 429 or 5xx with an appropriate delay is usually the correct behavior, since these failures are often transient and self-resolving.

## Retry-After Header Handling

Azure OpenAI includes a `Retry-After` header on 429 responses, expressed either as an integer number of seconds or, less commonly, as an HTTP-date string. A resilient client parses both formats defensively: attempt a direct integer conversion first, and fall back to HTTP-date parsing if that fails. Values should be clamped to a reasonable ceiling, such as ten seconds, to avoid a single misbehaving upstream response stalling the entire request pipeline. If the header is absent or malformed, the client should fall back to an exponential backoff schedule rather than failing immediately.

## Exponential Backoff Design

A typical backoff schedule doubles the delay after each attempt, starting from a small base value such as one second: 1s, 2s, 4s, 8s, capped at a maximum ceiling. The number of retry attempts should be bounded, commonly to three or four total tries for rate limit errors and fewer for server errors, since 5xx failures are less predictable and less likely to resolve quickly. Unbounded retries risk cascading load onto an already-struggling upstream service and can starve other requests waiting on the same connection pool.

## Distinguishing Retryable and Non-Retryable Failures

Not every failure should trigger a retry. Authentication and authorization errors indicate a configuration problem that will not resolve itself through waiting, so retrying them is wasted effort that delays surfacing the real issue to an operator. Malformed request bodies, invalid deployment names, and schema violations are client-side bugs, not transient conditions. Only 429 and 5xx responses represent genuinely transient failure modes where a bounded retry with backoff is the appropriate response.

## Operational Recommendations

Production deployments should log the retry count, the parsed delay value, and the eventual outcome for every retried request, enabling operators to distinguish between occasional rate-limit pressure and sustained capacity exhaustion. Dashboards tracking 429 rate over time provide an early warning signal before user-facing latency degrades. When sustained 429 rates are observed, the correct remediation is usually to request a TPM/RPM quota increase from Azure support or to introduce client-side request throttling ahead of the Azure gateway, rather than tightening retry logic further.
