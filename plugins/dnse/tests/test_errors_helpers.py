"""Direct unit tests for the module-private helpers of ``errors.py``.

``classify()`` (tested in ``test_errors.py``) only exercises these through a handful
of representative bodies; this file drives ``code_of``/``_message_of``/``_retry_after``
directly with the boundary shapes ``classify`` never happens to hit (missing keys,
non-dict bodies, falsy/non-numeric ``retryAfter``).
"""
import pytest

from pynecore_dnse.errors import DEFAULT_RETRY_AFTER, code_of, _message_of, _retry_after


# --- code_of -----------------------------------------------------------------

@pytest.mark.parametrize("body, want", [
    ({"code": "INVALID_PRICE"}, "INVALID_PRICE"),
    ({"code": ""}, ""),
    ({"code": None}, ""),
    ({}, ""),
    ("not a dict", ""),
    (None, ""),
    ([], ""),
    (42, ""),
], ids=[
    "dict-with-code", "dict-empty-code", "dict-none-code", "dict-no-code-key",
    "str-body", "none-body", "list-body", "int-body",
])
def __test_code_of__(body, want):
    got = code_of(body)
    assert got == want, f"code_of({body!r}) = {got!r}, want {want!r}"
    assert isinstance(got, str), "code_of must always return a str, never None"


# --- _message_of ---------------------------------------------------------------

@pytest.mark.parametrize("body, want", [
    ({"message": "bad price", "error": "ignored"}, "bad price"),
    ({"error": "server exploded"}, "server exploded"),
    ({}, ""),
    ({"message": None, "error": "fallback wins"}, "fallback wins"),
    ({"message": "", "error": "falsy-message-falls-back"}, "falsy-message-falls-back"),
    ("  raw string body  ", "raw string body"),
    (None, ""),
    ([1, 2, 3], ""),
    (7, ""),
], ids=[
    "message-wins-over-error", "falls-back-to-error", "dict-neither-key",
    "none-message-falls-back", "empty-message-falls-back", "str-body-stripped",
    "none-body", "list-body", "int-body",
])
def __test_message_of__(body, want):
    got = _message_of(body)
    assert got == want, f"_message_of({body!r}) = {got!r}, want {want!r}"
    assert isinstance(got, str), "_message_of must always return a str, never None"


def __test_message_of_truncates_long_string_body__():
    long_body = "x" * 250

    got = _message_of(long_body)

    assert len(got) == 200, f"expected truncation to 200 chars, got length {len(got)}"
    assert got == "x" * 200, "truncated text must be a prefix of the original body"


# --- _retry_after ----------------------------------------------------------------

@pytest.mark.parametrize("body, want", [
    ({"retryAfter": 12}, 12.0),
    ({"retryAfter": "7.5"}, 7.5),
    ({"retry_after": 3}, 3.0),
    ({"X-RateLimit-Reset": "9"}, 9.0),
    ({"retryAfter": 1, "retry_after": 2, "X-RateLimit-Reset": 3}, 1.0),
    ({"retryAfter": "soon", "retry_after": 4}, 4.0),
    ({"retryAfter": "soon"}, DEFAULT_RETRY_AFTER),
    ({"retryAfter": None}, DEFAULT_RETRY_AFTER),
    ({"retryAfter": 0}, DEFAULT_RETRY_AFTER),
    ({}, DEFAULT_RETRY_AFTER),
    (None, DEFAULT_RETRY_AFTER),
    ("not a dict", DEFAULT_RETRY_AFTER),
], ids=[
    "retryAfter-int", "retryAfter-numeric-str", "retry_after-key",
    "x-ratelimit-reset-key", "retryAfter-takes-priority",
    "non-numeric-falls-through-to-next-key", "non-numeric-only-key-uses-default",
    "none-value-uses-default", "falsy-zero-uses-default", "empty-dict-uses-default",
    "none-body-uses-default", "non-dict-body-uses-default",
])
def __test_retry_after__(body, want):
    got = _retry_after(body)
    assert got == want, f"_retry_after({body!r}) = {got!r}, want {want!r}"
    assert isinstance(got, float), "_retry_after must always return a float"
