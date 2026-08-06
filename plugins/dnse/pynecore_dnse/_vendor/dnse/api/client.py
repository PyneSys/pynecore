#!/usr/bin/env python3
import json
import os
from urllib import parse
import urllib3


from .common import build_signature, get_api_version, get_date_header_name


class DNSEClient:
    def __init__(
            self,
            api_key,
            api_secret,
            base_url="https://openapi.dnse.com.vn",
            algorithm="hmac-sha256",
            hmac_nonce_enabled=True,
            api_version=None,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._algorithm = algorithm
        self._hmac_nonce_enabled = hmac_nonce_enabled
        self._api_version = api_version or get_api_version()

        # Tạo PoolManager 1 lần duy nhất, tái sử dụng suốt vòng đời object
        self._http = urllib3.PoolManager(
            num_pools=10,           # Số lượng connection pools
            maxsize=10,             # Số connections tối đa mỗi pool
            block=False,            # Không block khi pool đầy
            timeout=urllib3.Timeout(connect=30.0, read=60.0),
            cert_reqs = 'CERT_NONE',  # Không yêu cầu certificate
            assert_hostname = False  # Không kiểm tra hostname
        )

    def get_accounts(self, dry_run=False):
        return self._request("GET", "/accounts", dry_run=dry_run)

    def get_balances(self, account_no, dry_run=False):
        return self._request("GET", f"/accounts/{account_no}/balances", dry_run=dry_run)

    def get_loan_packages(self, account_no, market_type, symbol=None, dry_run=False):
        query = {"marketType": market_type}
        if symbol:
            query["symbol"] = symbol
        return self._request(
            "GET",
            f"/accounts/{account_no}/loan-packages",
            query=query,
            dry_run=dry_run,
        )

    def get_positions(self, account_no, market_type, dry_run=False):
        return self._request(
            "GET",
            f"/accounts/{account_no}/positions",
            query={"marketType": market_type},
            dry_run=dry_run,
        )

    def get_position_by_id(self, market_type, position_id, version=None, dry_run=False):
        return self._request(
            "GET",
            f"/positions/{position_id}",
            query={"marketType": market_type},
            version=version,
            dry_run=dry_run,
        )

    def get_position_pnl_configs(self, market_type, position_id, version=None, dry_run=False):
        return self._request(
            "GET",
            f"/positions/{position_id}/pnl-configs",
            query={"marketType": market_type},
            version=version,
            dry_run=dry_run,
        )

    def post_position_pnl_configs(self, market_type, position_id, payload, trading_token, version=None, dry_run=False):
        headers = {"trading-token": trading_token}
        return self._request(
            "POST",
            f"/positions/{position_id}/pnl-configs",
            query={"marketType": market_type},
            body=payload,
            headers=headers,
            version=version,
            dry_run=dry_run,
        )

    def get_orders(
            self,
            account_no,
            market_type,
            order_category=None,
            page_index=None,
            page_size=None,
            dry_run=False,
    ):
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        if page_index is not None:
            query["pageIndex"] = page_index
        if page_size is not None:
            query["pageSize"] = page_size
        return self._request(
            "GET",
            f"/accounts/{account_no}/orders",
            query=query,
            dry_run=dry_run,
        )

    def get_order_detail(self, account_no, order_id, market_type, order_category=None, dry_run=False):
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        return self._request(
            "GET",
            f"/accounts/{account_no}/orders/{order_id}",
            query=query,
            dry_run=dry_run,
        )

    def get_execution_detail(self, account_no, order_id, market_type, order_category="NORMAL", dry_run=False):
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        return self._request(
            "GET",
            f"/accounts/{account_no}/executions/{order_id}",
            query=query,
            dry_run=dry_run,
        )

    def get_order_history(
            self,
            account_no,
            market_type,
            from_date=None,
            to_date=None,
            page_size=None,
            page_index=None,
            dry_run=False,
    ):
        query = {"marketType": market_type}
        if from_date:
            query["from"] = from_date
        if to_date:
            query["to"] = to_date
        if page_size is not None:
            query["pageSize"] = page_size
        if page_index is not None:
            query["pageIndex"] = page_index
        return self._request(
            "GET",
            f"/accounts/{account_no}/orders/history",
            query=query,
            dry_run=dry_run,
        )

    def get_corporate_action_history(
            self,
            account_no,
            symbol=None,
            ca_type=None,
            ca_status=None,
            page_index=None,
            page_size=None,
            dry_run=False,
    ):
        query = {}
        if symbol is not None:
            query["symbol"] = symbol
        if ca_type is not None:
            query["caType"] = ca_type
        if ca_status is not None:
            query["caStatus"] = ca_status
        if page_index is not None:
            query["pageIndex"] = page_index
        if page_size is not None:
            query["pageSize"] = page_size
        return self._request(
            "GET",
            f"/accounts/{account_no}/corporate-action-history",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_ppse(self, account_no, market_type, symbol, price, loan_package_id, dry_run=False):
        return self._request(
            "GET",
            f"/accounts/{account_no}/ppse",
            query={
                "marketType": market_type,
                "symbol": symbol,
                "price": str(price),
                "loanPackageId": str(loan_package_id),
            },
            dry_run=dry_run,
        )

    def get_security_definition(self, symbol, board_id=None, dry_run=False):
        query = {}
        if board_id:
            query["boardId"] = board_id
        return self._request(
            "GET",
            f"/price/{symbol}/secdef",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_ohlc(self, bar_type, query=None, dry_run=False):
        request_query = dict(query or {})
        request_query["type"] = bar_type
        return self._request(
            "GET",
            "/price/ohlc",
            query=request_query,
            dry_run=dry_run,
        )

    def get_trades(self, symbol, board_id=None, from_date=None, to_date=None, limit=None, order = None, next_page_token=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        if from_date is not None:
            query["from"] = from_date
        if to_date is not None:
            query["to"] = to_date
        if limit is not None:
            query["limit"] = limit
        if order is not None:
            query["order"] = order
        if next_page_token is not None:
            query["nextPageToken"] = next_page_token
        return self._request(
            "GET",
            f"/price/{symbol}/trades",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_expected_price(self, symbol, board_id=None, from_date=None, to_date=None, limit=None, order = None, next_page_token=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        if from_date is not None:
            query["from"] = from_date
        if to_date is not None:
            query["to"] = to_date
        if limit is not None:
            query["limit"] = limit
        if order is not None:
            query["order"] = order
        if next_page_token is not None:
            query["nextPageToken"] = next_page_token
        return self._request(
            "GET",
            f"/price/{symbol}/expected-price",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_quotes(self, symbol, board_id=None, from_date=None, to_date=None, limit=None, order = None, next_page_token=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        if from_date is not None:
            query["from"] = from_date
        if to_date is not None:
            query["to"] = to_date
        if limit is not None:
            query["limit"] = limit
        if order is not None:
            query["order"] = order
        if next_page_token is not None:
            query["nextPageToken"] = next_page_token
        return self._request(
            "GET",
            f"/price/{symbol}/quotes",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_foreign_trading(self, symbol, board_id=None, from_date=None, to_date=None, limit=None, order = None, next_page_token=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        if from_date is not None:
            query["from"] = from_date
        if to_date is not None:
            query["to"] = to_date
        if limit is not None:
            query["limit"] = limit
        if order is not None:
            query["order"] = order
        if next_page_token is not None:
            query["nextPageToken"] = next_page_token
        return self._request(
            "GET",
            f"/price/{symbol}/foreign-trading",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_instruments(self, symbol=None, market_id=None, security_group_id=None, index_name=None, limit=None, page=None, version=None, dry_run=False):
        query = {}
        if symbol is not None:
            query["symbol"] = symbol
        if market_id is not None:
            query["marketId"] = market_id
        if security_group_id is not None:
            query["securityGroupId"] = security_group_id
        if index_name is not None:
            query["indexName"] = index_name
        if limit is not None:
            query["limit"] = limit
        if page is not None:
            query["page"] = page
        return self._request(
            "GET",
            "/market/instruments",
            query=query if query else None,
            version=version,
            dry_run=dry_run,
        )

    def get_latest_trade(self, symbol, board_id=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        return self._request(
            "GET",
            f"/price/{symbol}/trades/latest",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_latest_quote(self, symbol, board_id=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        return self._request(
            "GET",
            f"/price/{symbol}/quotes/latest",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_close_price(self, symbol, board_id=None, dry_run=False):
        query = {}
        if board_id is not None:
            query["boardId"] = board_id
        return self._request(
            "GET",
            f"/price/{symbol}/close",
            query=query if query else None,
            dry_run=dry_run,
        )

    def get_working_dates(self, dry_run=False):
        return self._request(
            "GET",
            f"/market/working-dates",
            dry_run=dry_run,
        )

    def get_list_care_by(self, dry_run=False):
        return self._request(
            "GET",
            f"/brokers/accounts/care-by",
            dry_run=dry_run,
        )

    def get_lastest_session(self, tsc_prod_grp_id=None, board_id=None, dry_run=False):
        query = {}
        if tsc_prod_grp_id is not None:
            query["tscProdGrpId"] = tsc_prod_grp_id
        if board_id is not None:
            query["boardId"] = board_id
        return self._request(
            "GET",
            f"/market/trading-session",
            query=query if query else None,
            dry_run=dry_run,
        )

    def post_order(self, account_no, market_type, payload, trading_token, order_category="NORMAL", version=None, dry_run=False):
        headers = {"trading-token": trading_token}
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        return self._request(
            "POST",
            f"/accounts/{account_no}/orders",
            query=query,
            body=payload,
            headers=headers,
            version=version,
            dry_run=dry_run,
        )

    def put_order(
        self,
        account_no,
        order_id,
        market_type,
        payload,
        trading_token,
        order_category=None,
        dry_run=False,
    ):
        headers = {"trading-token": trading_token}
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        return self._request(
            "PUT",
            f"/accounts/{account_no}/orders/{order_id}",
            query=query,
            body=payload,
            headers=headers,
            dry_run=dry_run,
        )

    def cancel_order(
        self,
        account_no,
        order_id,
        market_type,
        trading_token,
        order_category=None,
        dry_run=False,
    ):
        headers = {"trading-token": trading_token}
        query = {"marketType": market_type}
        if order_category:
            query["orderCategory"] = order_category
        return self._request(
            "DELETE",
            f"/accounts/{account_no}/orders/{order_id}",
            query=query,
            headers=headers,
            dry_run=dry_run,
        )

    def create_trading_token(self, otp_type, passcode, dry_run=False):
        return self._request(
            "POST",
            "/registration/trading-token",
            body={"otpType": otp_type, "passcode": passcode},
            dry_run=dry_run,
        )

    def send_email_otp(self, dry_run=False):
        return self._request(
            "POST",
            "/registration/send-email-otp",
            dry_run=dry_run,
        )

    def close_position(self, position_id, market_type, trading_token, version=None, dry_run=False):
        headers = {"trading-token": trading_token}
        query = {"marketType": market_type}
        return self._request(
            "POST",
            f"/positions/{position_id}/close",
            query=query,
            headers=headers,
            version=version,
            dry_run=dry_run,
        )

    def _request(self, method, path, query=None, body=None, headers=None, version=None, dry_run=False):
        debug = os.getenv("DEBUG", "").lower() == "true"
        url = self._build_url(path, query)
        date_value, signature_header_value = self._signature_headers(method, path)
        date_header_name = get_date_header_name()

        # Build headers dict (per-request `version` overrides the client default)
        req_headers = {
            date_header_name: date_value,
            "X-Signature": signature_header_value,
            "x-api-key": self._api_key,
            "version": version or self._api_version,
        }

        if body is not None:
            req_headers["Content-Type"] = "application/json"

        if headers:
            req_headers.update(headers)

        # Prepare body data
        data = None
        if body is not None:
            data = json.dumps(body)

        if debug or dry_run:
            prefix = "DRY RUN" if dry_run else "DEBUG"
            print(f"{prefix} url:", url)
            print(f"{prefix} method:", method)
            print(f"{prefix} query_params:", query or {})
            print(f"{prefix} headers:", req_headers)
            print(f"{prefix} body:", body)

        if dry_run:
            return None, None

        try:
            resp = self._http.request(
                method,
                url,
                body=data,
                headers=req_headers,
            )
            body_text = resp.data.decode("utf-8")
            return resp.status, body_text
        except urllib3.exceptions.HTTPError as err:
            if hasattr(err, 'response') and err.response:
                body_text = err.response.data.decode("utf-8") if err.response.data else ""
                return err.response.status, body_text
            raise

    def _build_url(self, path, query):
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"
        return url

    def _signature_headers(self, method, path):
        date_value = self._date_header()
        nonce = None
        if self._hmac_nonce_enabled:
            import uuid

            nonce = uuid.uuid4().hex

        headers_list, signature = build_signature(
            self._api_secret,
            method,
            path,
            date_value,
            self._algorithm,
            nonce=nonce,
            header_name=get_date_header_name(),
        )
        signature_header_value = (
            f'Signature keyId="{self._api_key}",algorithm="{self._algorithm}",'
            f'headers="{headers_list}",signature="{signature}"'
        )
        if nonce:
            signature_header_value += f',nonce="{nonce}"'
        return date_value, signature_header_value

    def _date_header(self):
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
