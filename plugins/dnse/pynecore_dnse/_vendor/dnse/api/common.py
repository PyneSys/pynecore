#!/usr/bin/env python3
import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from urllib import parse, request
from uuid import uuid4

DEFAULT_API_VERSION = "2026-07-23"


def get_date_header_name():
    return os.getenv("DATE_HEADER", "Date")


def get_api_version():
    return os.getenv("DNSE_API_VERSION") or DEFAULT_API_VERSION


def build_signature(secret, method, path, date_value, algorithm, nonce=None, header_name=None):
    header_name = header_name or get_date_header_name()
    header_key = header_name.lower()
    headers = f"(request-target) {header_key}"
    signature_string = (
        f"(request-target): {method.lower()} {path}\n" f"{header_key}: {date_value}"
    )
    if nonce:
        signature_string += f"\nnonce: {nonce}"

    if algorithm == "hmac-sha256":
        digestmod = hashlib.sha256
    elif algorithm == "hmac-sha384":
        digestmod = hashlib.sha384
    elif algorithm == "hmac-sha512":
        digestmod = hashlib.sha512
    else:
        digestmod = hashlib.sha1

    mac = hmac.new(secret.encode("utf-8"), signature_string.encode("utf-8"), digestmod)
    encoded = base64.b64encode(mac.digest()).decode("utf-8")
    escaped = parse.quote(encoded, safe="")

    return headers, escaped


def send_signed_request(
    url,
    method,
    headers,
    body,
    api_key,
    api_secret,
    algorithm="hmac-sha256",
    hmac_nonce_enabled=True,
):
    debug = os.getenv("DEBUG", "").lower() == "true"
    headers = dict(headers or {})
    headers.setdefault("version", get_api_version())
    parsed = parse.urlparse(url)
    path = parsed.path
    date_value = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    date_header_name = get_date_header_name()

    nonce = uuid4().hex if hmac_nonce_enabled else None
    headers_list, signature = build_signature(
        api_secret,
        method,
        path,
        date_value,
        algorithm,
        nonce=nonce,
        header_name=date_header_name,
    )
    signature_header_value = (
        f'Signature keyId="{api_key}",algorithm="{algorithm}",'
        f'headers="{headers_list}",signature="{signature}"'
    )
    if nonce:
        signature_header_value += f',nonce="{nonce}"'

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    req = request.Request(url, data=data, method=method)
    req.add_header(date_header_name, date_value)
    req.add_header("X-Signature", signature_header_value)

    for key, value in headers.items():
        req.add_header(key, value)

    if debug:
        query_params = parse.parse_qs(parsed.query)
        print("DEBUG url:", url)
        print("DEBUG method:", method)
        print("DEBUG query_params:", query_params)
        print("DEBUG headers:", dict(req.header_items()))
        print("DEBUG body:", body)

    try:
        with request.urlopen(req) as resp:
            body_text = resp.read().decode("utf-8")
            print(body_text)
    except request.HTTPError as err:
        body_text = err.read().decode("utf-8") if err.fp else ""
        print(f"HTTP {err.code} {err.reason}")
        if body_text:
            print(body_text)
