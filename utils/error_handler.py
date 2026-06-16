import logging

logger = logging.getLogger("api.errors")


def friendly_error(exc: Exception) -> str:
    """
    Converts internal pipeline exceptions into short, user-facing messages.
    The original exception is always logged before this is called so no
    diagnostic detail is lost — we just don't expose it to the browser.
    """
    name = type(exc).__name__
    msg = str(exc)

    # Groq / OpenAI-compatible rate-limit (HTTP 429)
    if name == "RateLimitError" or (
        "429" in msg and ("rate_limit" in msg.lower() or "tpm" in msg.lower() or "tokens per minute" in msg.lower())
    ):
        return (
            "The AI model is temporarily rate-limited due to high token usage. "
            "Please wait a few seconds and try again."
        )

    # API key / authentication failure (HTTP 401)
    if name == "AuthenticationError" or (
        "401" in msg and "authentication" in msg.lower()
    ):
        return "AI service authentication failed. The API key may be invalid or expired."

    # Service unavailable (HTTP 503 / 502)
    if "503" in msg or "502" in msg or name == "APIStatusError":
        return "The AI service is temporarily unavailable. Please try again in a moment."

    # Network / connection issues
    if name in ("ConnectError", "TimeoutError", "ConnectionError") or (
        "timeout" in msg.lower() or "connection" in msg.lower()
    ):
        return "Could not reach the AI service. Please check your connection and try again."

    # Generic fallback — never expose raw exception strings to the UI
    return "An error occurred while processing your request. Please try again."
