"""
Run identity — logical and physical keys for a broker run.

A *run* is one concrete execution of a Pine strategy against a live broker
account for a specific (symbol, timeframe, optional label) combination.
Two keys exist for different purposes:

- **``run_id``** — humán-olvasható, determinisztikus stream-azonosító. Azonos
  strategy + account + symbol + timeframe + label kombináció mindig
  ugyanazt adja. Historikus lekérdezéshez (*"mutasd ennek a botnak az
  összes múltbeli futását"*) ez a kulcs.
- **``run_instance_id``** — fizikai, autoincrement INTEGER, egyedi minden
  élesben elindított futtatásra. FK-kulcs a storage összes táblájában.
  A storage tölti ki, ez a modul nem foglalkozik vele.

A ``RunIdentity`` ennek az egésznek a konstrukciós inputja: a strategy runner
a CLI-ből + plugin auth-ból összerakja, átadja a storage-nak, és visszakap
egy ``RunContext``-et a ``run_instance_id``-vel.

Miért külön modul az ``idempotency.py``-tól: az idempotency a
``client_order_id`` bit-szintű formáját kezeli (broker-protokoll), a
``RunIdentity`` a run-szintű identitást (PyneCore-belső). Két elkülönülő
absztrakciós réteg, egy modulban összemosva átláthatatlan lenne.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Final

from pynecore.core.broker.idempotency import RUN_TAG_WIDTH, _to_base36

__all__ = [
    'RunIdentity',
]

# 20 bits → 4 base36 chars, ugyanaz a tartomány mint az eredeti
# ``make_run_tag``-é. A bővebb input csak a collision-space-t javítja, a
# kimeneti formátum (``{run_tag}-{pid}-...``) változatlan.
_RUN_TAG_BITS: Final[int] = 20
_RUN_TAG_MASK: Final[int] = (1 << _RUN_TAG_BITS) - 1


@dataclass(frozen=True)
class RunIdentity:
    """Egy konkrét bot-futtatás identitása a storage előtt.

    :param strategy_id: A script-fájl stem-je (pl. ``"ema_cross"``). A
        kódbázis logikai neve, NEM verzió-hash — a forráskód-változásokat
        a ``run_tag`` ragadja meg a ``make_run_tag`` hash-bemenetén
        keresztül.
    :param symbol: Kereskedett instrumentum (pl. ``"EURUSD"``).
    :param timeframe: TradingView-formátum (pl. ``"60"``, ``"1D"``).
    :param account_id: Plugin-qualified broker-account azonosító, pl.
        ``"capitalcom-demo-1234567"``. A plugin sync ``.account_id``
        property-je adja, az autentikáció során populált. Hiányzó
        account esetén ``"default"``.
    :param label: Opcionális user-override (CLI ``--run-label``). Ugyanaz
        a (strategy_id, symbol, timeframe, account_id) kombináció több
        példányának megkülönböztetésére. Default: ``None``.
    """
    strategy_id: str
    symbol: str
    timeframe: str
    account_id: str
    label: str | None = None

    @property
    def run_id(self) -> str:
        """Humán-olvasható logikai kulcs.

        Formátum: ``"{strategy_id}@{account_id}:{symbol}:{timeframe}"``
        vagy label megadása esetén ``"...#{label}"``. A külön ``#``
        szeparátor azért kell, mert a ``:`` a timeframe után fordulhat
        elő (pl. ``"1D"``), a ``#`` viszont a CLI flagen keresztül
        érkező label-ben sem legális (a CLI validál).
        """
        base = f"{self.strategy_id}@{self.account_id}:{self.symbol}:{self.timeframe}"
        return f"{base}#{self.label}" if self.label else base

    def make_run_tag(self, script_source: str) -> str:
        """4-char base36 session tag a client_order_id-hez.

        Az idempotency formulának (``{run}-{pid}-{bar}-{k}{r}``) 4 karakter
        áll rendelkezésre erre a mezőre; a bit-térfogat tehát 20 bit
        (~1M slot). Determinisztikus: azonos input mindig azonos tag.

        Miért több input, mint a régi ``make_run_tag(script_source)``:
        ugyanaz a script két timeframe-en egyszerre futva azonos tag-et
        adott, ami idempotency-collision-t eredményezett (azonos
        ``client_order_id``-k). A ``strategy_id`` / ``symbol`` /
        ``timeframe`` / ``account`` / ``label`` bevonásával a
        collision-space kiterjed a realisztikus run-dimenziókra.
        ``strategy_id`` külön is kell: két külön script, azonos forrással
        (copy-paste) ugyanarra a (symbol, tf, account) kombinációra
        egyébként azonos tag-et kapna, és a brokernél duplikált
        ``client_order_id``-kat generálna.

        A JSON-serializált input determinisztikus stringifikációt ad
        (pipe/quote/unicode edge-case-mentes).

        :param script_source: A Pine-script forráskódja, ahogy a runner
            beolvasta.
        :return: Pontosan 4 karakter lower-case base36.
        """
        payload = json.dumps(
            [
                self.strategy_id,
                script_source,
                self.symbol,
                self.timeframe,
                self.account_id,
                self.label or "",
            ],
            ensure_ascii=True,
            sort_keys=False,
        )
        digest = hashlib.sha256(payload.encode('ascii')).digest()
        # 20 bit → 4 char base36. A 3 byte-os slice bőven lefedi,
        # utána AND-maszkolunk.
        value = int.from_bytes(digest[:3], 'big') & _RUN_TAG_MASK
        return _to_base36(value, width=RUN_TAG_WIDTH)
