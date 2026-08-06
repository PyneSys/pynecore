"""DNSE plugins for PyneCore — v2.

v1 (a software-watch design over the vendored ``dnse-py`` SDK) has been removed.
v2 is rebuilt on DNSE's official **openapi-sdk 2.0.0** and uses DNSE's **native
conditional orders** (``orderCategory=STOP|OCO`` on the account-scoped
``/accounts/{accountNo}/orders`` endpoints, ``version >= 2026-07-23``).

See ``docs/dnse-openapi-documentation/`` for the mirrored API + SDK docs.
"""
