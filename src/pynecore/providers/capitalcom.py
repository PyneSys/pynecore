from typing import Callable, cast
import sys

# Import the override decorator for Python 3.12+
if sys.version_info >= (3, 12):
    from typing import override
else:
    # An empty decorator for Python 3.11 and below
    def override(func):
        return func
from datetime import datetime, time, UTC, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from functools import lru_cache

from .provider import Provider

from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from ..types.ohlcv import OHLCV

__all__ = ['CapitalComProvider']

URL = 'https://api-capital.backend-capital.com'
URL_DEMO = 'https://demo-api-capital.backend-capital.com'

ENDPOINT_PREFIX = '/api/v1/'

TIMEFRAMES = {
    # TradingView -> Capital.com
    '1': 'MINUTE',
    '5': 'MINUTE_5',
    '15': 'MINUTE_15',
    '30': 'MINUTE_30',
    '60': 'HOUR',
    '240': 'HOUR_4',
    '1D': 'DAY',
    '1W': 'WEEK'
}

TIMEFRAMES_INV = {v: k for k, v in TIMEFRAMES.items()}

TYPES = {
    'CURRENCIES': 'forex',
    'CRYTOCURRENCIES': 'crypto',
    'SHARES': 'stock',
    'INDICES': 'index',
}


def encrypt_password(password: str, encryption_key: str, timestamp: int | None = None):
    from time import time as epoch
    try:
        from base64 import standard_b64encode, standard_b64decode
        from Crypto.PublicKey import RSA
        from Crypto.Cipher import PKCS1_v1_5
    except ImportError:
        raise ImportError('The "pycryptodome" package is required for Capital.com provider. Please install it by '
                          'running `pip install pycryptodome`')

    if timestamp is None:
        timestamp = int(epoch())
    payload = password + '|' + str(timestamp)
    payload = standard_b64encode(payload.encode('ascii'))
    public_key = RSA.importKey(standard_b64decode(encryption_key.encode('ascii')))
    cipher = PKCS1_v1_5.new(public_key)
    ciphertext = standard_b64encode(cipher.encrypt(payload)).decode()
    return ciphertext


class CaptialComError(ValueError):
    ...


class CapitalComProvider(Provider):
    """
    Capital.com provider
    """

    timezone = 'US/Eastern'
    config_keys = {
        '# If it is a demo account': '',
        'demo': False,
        '# These are required for Capital.com. You can get them from the Capital.com API settings.': '',
        'user_email': '',
        'api_key': '',
        'api_password': ''
    }

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        """
        Convert Capital.com timeframe format to TradingView format.

        :param timeframe: Timeframe in Capital.com format (e.g. "MINUTE", "MINUTE_5", "HOUR", "DAY")
        :return: Timeframe in TradingView format (e.g. "1", "5", "60", "1D")
        :raises ValueError: If timeframe format is invalid
        """
        try:
            return TIMEFRAMES_INV[timeframe.upper()]
        except KeyError:
            raise ValueError(f"Invalid Capital.com timeframe format: {timeframe}")

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        """
        Convert TradingView timeframe format to Capital.com format.

        :param timeframe: Timeframe in TradingView format (e.g. "1", "5", "60", "1D")
        :return: Timeframe in Capital.com format (e.g. "MINUTE", "MINUTE_5", "HOUR", "DAY")
        :raises ValueError: If timeframe format is invalid
        """
        try:
            return TIMEFRAMES[timeframe]
        except KeyError:
            raise ValueError(f"Unsupported timeframe for Capital.com: {timeframe}")

    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlv_dir: Path | None = None, config_dir: Path | None = None):
        """
        :param symbol: The symbol to get data for
        :param timeframe: The timeframe to get data for in TradingView fmt
        :param ohlv_dir: The directory to save OHLV data
        :param config_dir: The directory to read the config file from
        """
        super().__init__(symbol=symbol, timeframe=timeframe, ohlv_dir=ohlv_dir, config_dir=config_dir)
        self.security_token = None
        self.cst_token = None
        self.session_data = {}

    # Basic API calls

    def __call__(self, endpoint: str, *, data: dict = None, method='post', _level=0) -> dict | list[dict]:
        """
        Call General API endpoints
        """
        from json import JSONDecodeError
        try:
            import httpx
        except ImportError:
            raise ImportError('The "httpx" package is required for Capital.com provider. Please install it by '
                              'running `pip install httpx`')

        headers = {'X-CAP-API-KEY': self.config['api_key']}
        if self.security_token:
            headers['X-SECURITY-TOKEN'] = self.security_token
        if self.cst_token:
            headers['CST'] = self.cst_token

        method = method.lower()
        params = dict(headers=headers, timeout=50.0)
        if method == 'get':
            params['params'] = data
        elif method in ('post', 'put'):
            params['json'] = data

        url = URL_DEMO if self.config['demo'] else URL
        url += ENDPOINT_PREFIX + endpoint

        res: httpx.Response = getattr(httpx, method)(url, **params)
        try:
            dict_res = res.json()
        except JSONDecodeError:
            raise CaptialComError(f"JSON Error: {res.text}")

        if res.is_error:
            # Relogin/autologin if missing token
            if dict_res['errorCode'] in ('error.security.client-token-missing', 'error.null.client.token') \
                    and self.config['user_email'] and self.config['api_password'] and _level < 3:
                # Create new session
                self.create_session()
                # Retry original request
                return self(endpoint=endpoint, data=data, method=method, _level=_level + 1)
            raise CaptialComError(f"API error occured: {dict_res['errorCode']}")

        try:
            self.security_token = res.headers['X-SECURITY-TOKEN']
        except KeyError:
            pass
        try:
            self.cst_token = res.headers['CST']
        except KeyError:
            pass

        return dict_res

    def create_session(self):
        """
        Create Session
        """
        res: dict = self('session/encryptionKey', method='get')
        encryption_key = res['encryptionKey']
        timestamp = res['timeStamp']
        user = self.config['user_email']
        api_password = self.config['api_password']
        password = encrypt_password(api_password, encryption_key, timestamp)
        self.session_data = self('session', data=dict(
            encryptedPassword=True,
            identifier=user,
            password=password
        ))

    ###

    def get_market_details(self, search_term: str = None, symbols: list[str] = None) -> dict:
        """
        Get and search market details
        """
        data = {}
        if search_term:
            data['searchTerm'] = search_term
        if symbols:
            data['epics'] = ','.join(symbols)
        res: dict = self('markets', data=data, method='get')
        return res

    @lru_cache(maxsize=1)
    def get_single_market_details(self) -> dict:
        """
        Get market details of a symbol
        """
        assert self.symbol is not None
        return cast(dict, self('markets/' + self.symbol, method='get'))

    def get_historical_prices(self, time_from: datetime = None, time_to: datetime = None, limit=1000) -> dict:
        """
        Get historical prices of market

        :param time_from: The start time (interpreted as UTC)
        :param time_to: The end time (interpreted as UTC)
        :param limit: The maximum number of candles to return
        """
        assert self.symbol is not None
        assert self.xchg_timeframe is not None
        params = {'resolution': self.xchg_timeframe, 'max': limit}
        if time_from is not None:
            params['from'] = time_from.isoformat()
        if time_to is not None:
            params['to'] = time_to.isoformat()
        res: dict = self('prices/' + self.symbol, data=params, method='get')
        return res

    @override
    def get_opening_hours_and_sessions(self) \
            -> tuple[list[SymInfoInterval], list[SymInfoSession], list[SymInfoSession]]:
        """
        Get opening hours and sessions of a symbol
        """
        from ..types.weekdays import Weekdays

        market_details = self.get_single_market_details()
        instrument = market_details['instrument']
        opening_hours = instrument['openingHours']

        # noinspection PyShadowingNames
        def timetz(t: time, tz: str) -> time:
            dt = datetime.now(ZoneInfo(tz))
            dt = dt.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond)
            dt = dt.astimezone(ZoneInfo(self.timezone))
            return dt.time()

        tz = opening_hours['zone']
        intervals = []
        session_starts = []
        session_ends = []

        for day in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']:
            ohs = opening_hours[day]
            day = Weekdays[day.capitalize()]
            for oh in ohs:
                oh = oh.replace('00:00', '').strip()
                if oh.startswith('-'):
                    t = timetz(time.fromisoformat(oh[2:]), tz)
                    intervals.append(SymInfoInterval(day=day.value, start=time(hour=0, minute=0), end=t))
                    session_ends.append(SymInfoSession(day=day.value, time=t))
                elif oh.endswith('-'):
                    t = timetz(time.fromisoformat(oh[:-2]), tz)
                    intervals.append(SymInfoInterval(day=day.value, start=t, end=time(hour=0, minute=0)))
                    session_starts.append(SymInfoSession(day=day.value, time=t))

        return intervals, session_starts, session_ends

    @override
    def get_list_of_symbols(self, *args, search_term: str = None) -> list[str]:
        """
        Get list of symbols

        :param search_term: Search term
        """
        res: dict = self.get_market_details(search_term=search_term)
        markets = [m['epic'] for m in res['markets']]
        markets.sort()
        return markets

    @override
    def update_symbol_info(self) -> SymInfo:
        """
        Update symbol info from the exchange
        """
        market_details = self.get_single_market_details()
        instrument = market_details['instrument']

        # Get opening hours and sessions
        opening_hours, session_starts, session_ends = self.get_opening_hours_and_sessions()

        dealing_rules = market_details['dealingRules']
        mintick = dealing_rules['minStepDistance']["value"]
        minmove = mintick
        pricescale = 1
        while minmove < 1.0:
            pricescale *= 10
            minmove *= 10

        # Download some data to get the average spread
        res = self.get_historical_prices()
        avg_spred_summ = 0.0
        for p in res['prices']:
            spread = abs(p['closePrice']['bid'] - p['closePrice']['ask'])
            avg_spred_summ += spread
        avg_spred = avg_spred_summ / len(res['prices'])

        return SymInfo(
            prefix=self.__class__.__name__.replace('Provider', '').upper(),
            description=instrument['name'],
            ticker=instrument['epic'],
            currency=instrument['currency'],
            basecurrency=instrument['symbol'].split('/')[0] if '/' in instrument['symbol'] else None,
            period=self.timeframe,
            type=TYPES[instrument['type']] if instrument['type'] in TYPES else 'other',
            mintick=mintick,
            pricescale=pricescale,
            minmove=minmove,
            pointvalue=instrument['lotSize'],
            timezone=self.timezone,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            # This is not found on TV, but it could be useful
            avg_spread=avg_spred,
        )

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None):
        """
        Download OHLV data

        :param time_from: The start time
        :param time_to: The end time
        :param on_progress: Optional callback to call on progress
        :param limit: Override the automatic chunk size (number of bars per API request)
        """

        # Shortcuts for the time_from and time_to
        tf = time_from.replace(tzinfo=None)
        tt = (time_to if time_to is not None else datetime.now(UTC)).replace(tzinfo=None)

        try:
            # Loop through the time range
            d = None
            while tf < tt:
                if on_progress:
                    on_progress(tf)

                res: dict = self.get_historical_prices(time_from=tf, limit=limit or 1000)
                if not res or not res['prices']:
                    break
                ps = res['prices']
                if len(ps) == 1 and d is not None:
                    break

                for p in ps:
                    t = datetime.fromisoformat(p['snapshotTimeUTC'])

                    # Filter wrong data, are not on TradingView :-/
                    if p['lastTradedVolume'] <= 1.0:
                        tf = t + timedelta(minutes=1)
                        continue

                    if t > tt:
                        raise StopIteration
                    ohlcv = OHLCV(
                        timestamp=int(t.timestamp()),
                        # Tradingview uses bidprice, not midprice
                        open=float(p['openPrice']['bid']),
                        high=float(p['highPrice']['bid']),
                        low=float(p['lowPrice']['bid']),
                        close=float(p['closePrice']['bid']),
                        volume=float(p['lastTradedVolume']),
                    )

                    self.save_ohlcv_data(ohlcv)
                    tf = t + timedelta(minutes=1)

        except CaptialComError:
            pass

        except StopIteration:
            pass

        if on_progress:
            on_progress(tt)
