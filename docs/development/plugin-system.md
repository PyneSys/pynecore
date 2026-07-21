<!--
---
weight: 1104
title: "Plugin System"
description: "How to create plugins for PyneCore"
icon: "extension"
date: "2026-03-30"
lastmod: "2026-07-19"
draft: false
toc: true
categories: ["Development"]
tags: ["plugins", "providers", "cli", "extensions", "entry-points"]
---
-->

# Plugin System

PyneCore uses a plugin architecture based on Python entry points.  Plugins can
provide data sources, add CLI commands, or extend existing commands with new
parameters — all discovered automatically at startup.

## Architecture

Every plugin registers under a single entry point group: `pyne.plugin`.  The
class hierarchy determines what a plugin can do:

```
Plugin (base)
├── ProviderPlugin           — Offline OHLCV data provider
│   └── LiveProviderPlugin   — WebSocket/streaming data (extends ProviderPlugin)
│       └── BrokerPlugin     — Order execution (extends LiveProviderPlugin)
└── CLIPlugin                — CLI subcommands and parameter hooks
```

`LiveProviderPlugin` inherits from `ProviderPlugin` — every live provider can also download
historical data.  See [Live Mode](../advanced/live-mode.md) for data-side details.

`BrokerPlugin` inherits from `LiveProviderPlugin` — an exchange that routes orders can
also deliver the live market data those orders trade against.  Order execution is handled
by dedicated per-exchange broker plugins (`pynesys-pynecore-bybit`,
`pynesys-pynecore-capitalcom`, `pynesys-pynecore-ctrader`) — not by standalone data
providers.

Multiple inheritance combines capabilities:

```python
class MyPlugin(ProviderPlugin, CLIPlugin):
    """A plugin that provides both data downloading and CLI commands."""
    ...
```

## Quick Start: Hello Plugin

A minimal plugin that adds `pyne hello greet` to the CLI.

### Project structure

```
pynecore-hello/
├── pyproject.toml
└── src/
    └── pynecore_hello/
        └── __init__.py
```

### pyproject.toml

```toml
[project]
name = "pynecore-hello"
version = "0.1.0"
description = "Hello World plugin for PyneCore"
dependencies = ["pynesys-pynecore[cli]>=6.5"]

# This is how PyneCore discovers the plugin automatically:
#   "hello" = the plugin name (used as `pyne hello ...`)
#   "pynecore_hello:HelloPlugin" = module:class to load
[project.entry-points."pyne.plugin"]
hello = "pynecore_hello:HelloPlugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The `[project.entry-points."pyne.plugin"]` section is the key: it tells Python
that when someone installs this package, a plugin named `hello` should be
registered, pointing to the `HelloPlugin` class.

### __init__.py

```python
import typer
from pynecore.core.plugin import CLIPlugin


class HelloPlugin(CLIPlugin):
    """Hello World plugin."""

    @staticmethod
    def cli() -> typer.Typer:
        app = typer.Typer(help="Hello World commands")

        @app.command()
        def greet(name: str = typer.Argument("World", help="Who to greet")):
            """Say hello."""
            typer.echo(f"Hello, {name}!")

        return app
```

### Install and run

```bash
pip install -e pynecore-hello/
pyne hello greet PyneCore
# Hello, PyneCore!
```

That's it.  The plugin is discovered automatically — no registration code, no
config files, no imports.  Just install and use.

## Plugin Capabilities

### CLIPlugin — Commands and Parameter Hooks

`CLIPlugin` provides two independent mechanisms:

#### 1. Subcommands via `cli()`

Return a Typer app to add a command group under `pyne <plugin_name>`:

```python
class FooPlugin(CLIPlugin):
    @staticmethod
    def cli() -> typer.Typer:
        app = typer.Typer(help="Foo commands")

        @app.command()
        def bar(name: str = typer.Argument("world")):
            """Do something."""
            typer.echo(f"Hello {name}")

        return app
```

This creates `pyne foo bar`.

#### 2. Parameter hooks via `cli_params()`

Inject flags into existing commands (`run` and `data download` are currently
pluggable; nested subcommands are addressed by their space-separated path,
e.g. `"data download"`):

```python
import click

class FooPlugin(CLIPlugin):
    @staticmethod
    def cli_params(command_name: str) -> list[click.Parameter]:
        if command_name == "run":
            return [
                click.Option(
                    ["--verbose", "-V"],
                    is_flag=True,
                    default=False,
                    help="Enable verbose output",
                ),
            ]
        return []
```

These parameters appear in `pyne run --help` and are parsed automatically.
The values are available via `ctx.plugin_params` in the command callback:

```python
# Inside the run command implementation
def run(ctx: typer.Context, script: Path = ..., data: Path = ...):
    verbose = ctx.plugin_params.get("verbose", False)
```

> **Note:** Use standard `click.Option` / `click.Argument` — these are the same
> objects you'd use in any Click application.  Typer-specific features like
> `rich_help_panel` are not available on injected parameters.

#### Conflict Detection

The plugin system prevents parameter collisions automatically:

- If a plugin tries to register `--from` but the `run` command already uses it →
  registration fails with a warning
- If two plugins both register `--verbose` → the second one fails with a warning
- If a plugin tries to use a built-in command name (`run`, `data`, `compile`,
  etc.) as its subcommand name → skipped with a warning

Both parameter names (`time_from`) and option strings (`--from`, `-f`) are
checked to prevent ambiguity.

### ProviderPlugin — Data Sources

Provides offline OHLCV data download capability.  Used by `pyne data download`.

```python
from datetime import datetime
from dataclasses import dataclass
from pynecore.core.plugin import ProviderPlugin, override
from pynecore.core.syminfo import SymInfo


@dataclass
class FooConfig:
    """Foo provider"""

    api_key: str = ""
    """API key for authentication"""

    use_sandbox: bool = False
    """Use sandbox environment"""


class FooProvider(ProviderPlugin[FooConfig]):
    Config = FooConfig

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        ...  # e.g. "1h" -> "60"

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        ...  # e.g. "60" -> "1h"

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        ...

    @override
    def update_symbol_info(self) -> SymInfo:
        ...  # fetch symbol metadata, including opening hours and sessions

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress=None, limit=None, with_extra=False):
        ...  # fetch bars and persist them via self.save_ohlcv_data(...)
```

Five abstract methods are required: the two timeframe converters (TradingView
format like `"60"` / `"1D"` ↔ the exchange-native format), `get_list_of_symbols`,
`update_symbol_info` (must return a fully populated `SymInfo`, including
`session_starts` / `session_ends` — populate them even for 24/7 markets), and
`download_ohlcv` (fetch the `time_from`..`time_to` window and write records
with `save_ohlcv_data`).

The `Config` dataclass is automatically turned into a self-healing TOML file
at `workdir/config/plugins/<name>.toml` — generated with all defaults commented
out, users uncomment and edit what they need:

```toml
# Foo provider

# API key for authentication
#api_key = ""

# Use sandbox environment
#use_sandbox = false
```

The Generic type parameter (`ProviderPlugin[FooConfig]`) gives your IDE
full type information on `self.config` — no more `object | None` warnings.

#### Multi-broker providers

A single provider plugin can serve many brokers/exchanges (CCXT reaches 100+). Declare it with the `multi_broker` class attribute and the provider string's broker segment is parsed for you:

```python
from pynecore.core.plugin import Broker


class FooProvider(ProviderPlugin[FooConfig]):
    Config = FooConfig
    multi_broker = True

    @classmethod
    @override
    def get_list_of_brokers(cls) -> list[Broker]:
        """Powers `pyne data download foo --list-brokers`."""
        return [Broker("broker-a", "Broker A"), Broker("broker-b")]
```

`Broker` is a small NamedTuple: `id` is the space-free selector used in the
provider string and the saved filename, `name` is an optional human-readable
display name.

With `multi_broker = True`, a provider string like `foo:BROKER:SYMBOL@TF` is split so the segment after the provider name becomes the broker. The broker is then **re-folded into the symbol** before it reaches your plugin — your `__init__` receives `"BROKER:SYMBOL"` and splits it as needed, exactly how providers have always received their symbol, so existing parsing keeps working. The optional `get_list_of_brokers()` classmethod powers `--list-brokers`; leave it unimplemented (the default raises `NotImplementedError`) when the broker set can't be enumerated.

### LiveProviderPlugin — Streaming Data

A `LiveProviderPlugin` is a `ProviderPlugin` that can also **stream bars in
real time** — this is what `pyne run --live` needs.  On top of the five
`ProviderPlugin` abstract methods it adds a connection lifecycle and a
blocking bar-watch call:

```python
from dataclasses import dataclass
from pynecore.core.plugin import LiveProviderPlugin, LiveProviderConfig, override
from pynecore.types.ohlcv import OHLCV


@dataclass
class FooLiveConfig(LiveProviderConfig):
    """Foo live provider"""

    api_key: str = ""
    """API key for authentication"""


class FooLiveProvider(LiveProviderPlugin[FooLiveConfig]):
    Config = FooLiveConfig

    # ... the five ProviderPlugin abstract methods (see above) ...

    @override
    async def connect(self) -> None:
        ...  # authenticate, open the WebSocket, subscribe

    @override
    async def disconnect(self) -> None:
        ...  # close the connection cleanly

    @property
    @override
    def is_connected(self) -> bool:
        ...

    @override
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        ...  # await the next update from the stream
```

The contract behind those four members:

- **Config base class.**  A live plugin's config subclasses
  `LiveProviderConfig`, which contributes the shared `symbol_map`
  translation table (TradingView-style symbol → exchange-native
  identifier).  Symbols not in the map fall back to `normalize_symbol()` —
  do not invent a plugin-specific translation knob.
- **`watch_ohlcv()` blocks until the next update** and returns an `OHLCV`
  with `is_closed=True` for a final bar, `False` for an intra-bar update.
  The framework calls it in a loop.
- **`connect()` is called again on every reconnect**, not just once.  The
  framework drives a full `disconnect()` → `connect()` cycle after a
  connection error or a stale feed, so `connect()` must re-initialize
  every piece of connection-scoped state (auth, server-side subscriptions,
  partially accumulated quote state).
- **Reconnection is retried indefinitely** with exponential backoff — the
  `reconnect_delay` class attribute doubles per failure up to
  `max_reconnect_delay`.  Do not add your own retry limit; a live bot must
  survive arbitrarily long outages.
- **Feed-liveness watchdog.**  When the transport looks alive but
  `watch_ohlcv` stays silent for `feed_timeout_bars` timeframe periods
  during an open session, the framework assumes a half-open socket / lost
  subscription and forces a reconnect cycle.  Only set the attribute to
  `None` when the venue's own liveness machinery genuinely covers the
  dead-feed case.
- **Optional hooks**: `on_disconnect()` / `on_reconnect()` for custom
  reconnect behavior, and `can_shutdown()` — polled every second during
  graceful shutdown; return `False` to delay exit while cleanup (closing
  positions, draining orders) is still in progress.

The async methods run in a background thread; the framework bridges them to
the synchronous `ScriptRunner` via a queue — the plugin never deals with
that plumbing.  For the user-facing behavior (historical → live transition,
intra-bar updates, `barstate`, CLI flags) see
[Live Mode](../advanced/live-mode.md).

### BrokerPlugin — Order Execution

A `BrokerPlugin` is a `LiveProviderPlugin` that can also **route orders** to an
exchange.  It receives high-level intents from the engine (`execute_entry`,
`execute_exit`, `execute_close`, `execute_cancel`) and translates them into
exchange-specific calls.  The engine handles idempotency, retry, and reconcile
— the plugin focuses on the actual REST/WebSocket wiring.

This section covers the scaffolding only. Before writing a real broker
plugin, read the
[Broker Plugin Authoring Guide](broker-plugin-authoring.md) — it is the
checklist of contract requirements (disappearance detection, incremental
fill reporting, client-id echo, override pairs, lifecycle order) that live
trading depends on and the type signatures alone do not show.

```python
from dataclasses import dataclass
from pynecore.core.plugin import LiveProviderConfig, override
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.models import (
    CapabilityLevel, DispatchEnvelope, ExchangeCapabilities,
    ExchangeOrder, ExchangePosition,
)


@dataclass
class FooBrokerConfig(LiveProviderConfig):
    """Foo exchange credentials."""
    api_key: str = ""
    api_secret: str = ""
    demo: bool = True


class FooBroker(BrokerPlugin[FooBrokerConfig]):
    Config = FooBrokerConfig

    @override
    async def connect(self) -> None:
        # Authenticate and populate self._account_id.
        # The account_id property later reads it back as a sync value.
        await self._authenticate()
        self._account_id = f"foo-{'demo' if self.config.demo else 'live'}-{self._login}"

    @override
    def get_capabilities(self) -> ExchangeCapabilities:
        # Every field is a CapabilityLevel (UNSUPPORTED / SOFTWARE /
        # PARTIAL_NATIVE / NATIVE), never a bool — see
        # pynecore.core.broker.models for the full struct.
        return ExchangeCapabilities(
            stop_order=CapabilityLevel.NATIVE,
            tp_sl_bracket=CapabilityLevel.NATIVE,
            reduce_only=CapabilityLevel.NATIVE,
        )

    @override
    async def execute_entry(self, envelope: DispatchEnvelope) -> list[ExchangeOrder]:
        ...

    @override
    async def get_position(self, symbol: str) -> ExchangePosition | None:
        ...
```

A broker config subclasses `LiveProviderConfig` — that is the bound of the
`BrokerPlugin` type parameter, and it contributes the `symbol_map` translation
table every live plugin shares.

#### Storage — `self.store_ctx`

Every broker plugin gets a `RunContext` wired in by `ScriptRunner` at startup
(`self.store_ctx`).  This is the single entry point for persistence — you do
**not** write your own JSONL, SQLite, or in-memory bookkeeping.  The
`RunContext` is backed by a shared `BrokerStore` (SQLite, WAL mode) at
`workdir/output/logs/broker.sqlite`, and it gives you:

- **Generic alias lookup.**  Exchange IDs that arrive later in the lifecycle
  (Capital.com `dealId`, IB `permId`, Bybit `orderLinkId`) are stored in the
  `order_refs` table.  Reverse lookup is a single indexed SELECT:

  ```python
  # When the exchange returns a durable ID, stash it as an alias.
  self.store_ctx.add_ref(client_order_id, 'exchange_order_id', exchange_id)

  # Later, when a fill event arrives with only the exchange ID, resolve it:
  row = self.store_ctx.find_by_ref('exchange_order_id', exchange_id)
  if row is not None:
      client_order_id = row.client_order_id
  ```

- **Audit log.**  Plugin-specific events (rate-limit hits, degraded protection,
  reconcile outcomes) go through `log_event`:

  ```python
  self.store_ctx.log_event(
      'rate_limit_hit',
      client_order_id=coid,
      payload={'retry_after_s': 1.5},
  )
  ```

- **Order state writes.**  The sync engine handles the canonical order
  lifecycle automatically.  Only touch `upsert_order` / `set_exchange_id` /
  `set_risk` if your plugin needs to record extra state the engine doesn't
  know about.

**Authentication and `account_id`.**  `BrokerPlugin.account_id` is a sync
property that returns `self._account_id`.  Your `connect()` (or the first
authenticating call) must populate `self._account_id` as a
**plugin-qualified** string, e.g. `"foo-demo-1234567"`.  The `ScriptRunner`
reads it once during startup to build the `run_id` — if the bot later
switches accounts on the broker UI, the stored `run_id` won't silently drift.

**Restart recovery.**  If the process is `SIGKILL`-ed or the host restarts,
the `runs` row is left with `ended_ts_ms IS NULL` but its heartbeat goes
stale.  The next startup's `open_run()` automatically closes stale rows
(heartbeat > 5 min) and logs a `stale_run_cleaned` event.  There is nothing
for the plugin to do here — recovery is built into the store.

#### `BrokerStore` schema — what gets stored where

A single SQLite file at `workdir/output/logs/broker.sqlite` is shared by
every bot process in the same workdir (WAL mode; one writer at a time, no
blocked readers).  Two identity keys share the tables:

- **`run_id`** — logical stream, the humanly recognizable identifier of
  a bot: `"{strategy}@{account}:{symbol}:{tf}[#label]"`.  Stable across
  restarts.
- **`run_instance_id`** — physical autoincrement integer, unique per
  process-level run.  Historical isolation.

| Table                   | Keyed by             | What it holds                                          |
|-------------------------|----------------------|--------------------------------------------------------|
| `runs`                  | `run_instance_id`    | Per-run metadata, heartbeat, lifecycle timestamps.     |
| `envelopes`             | `run_id`             | Sync engine envelope identity (cross-restart).         |
| `pending_verifications` | `run_id`             | Parked dispatches awaiting confirmation.               |
| `orders`                | `run_instance_id`    | Live order snapshot (+ plugin-specific `extras` JSON). |
| `order_refs`            | `run_instance_id`    | Generic alias lookup (broker IDs → `client_order_id`). |
| `events`                | `run_instance_id`    | Audit log (dispatch, fill, reconcile, stale-cleanup).  |

The `envelopes` and `pending_verifications` tables key on the **logical**
`run_id`, so a restarted bot picks up the same idempotency anchors.
Everything else keys on the **physical** `run_instance_id`, so historical
runs stay isolated.

## Offline Broker Conformance Lab

The broker conformance lab exercises a broker plugin against the real
`OrderSyncEngine`, `BrokerStore`, `RunContext`, and `BrokerPosition` without
connecting to a venue. It is deliberately opt-in: suites live in a top-level
`broker_lab/` directory, outside `tests/`, and are not collected by the normal
PyneCore or plugin test commands.

There are three distinct validation layers:

1. `validate_plugin_contract()` checks that required methods, capabilities,
   identity, and configuration are structurally coherent.
2. The offline conformance lab checks lifecycle, persistence, event ordering,
   idempotency, ownership, and modeled venue semantics deterministically.
3. A final live venue smoke checks facts that an offline model cannot establish,
   such as undocumented exchange behaviour, authentication, rate limits, and
   production network timing.

The lab can find duplicate dispatch after restart, replayed fills, stale-read
dispatch, cross-run adoption, incomplete protective coverage, illegal terminal
order resurrection, unbounded rejection retries, and quantity-grid residuals.
It cannot prove that an undocumented venue response is modeled correctly, nor
does it simulate price discovery, liquidity, matching priority, or outages in
the real network. Passing proves only the semantics encoded in the profile.

### Minimal profile and suite

PyneCore ships a copyable reference at `broker_lab/suite.py`. A plugin profile
implements `VenueProfile`; the usual pattern is to subclass the real broker and
replace only its lowest transport boundary:

```python
from typing import Any

from pynecore.testing.broker_lab import (
    ReferenceVenueProfile,
    Scenario,
    Step,
)
from pynecore_foo import FooBroker


class OfflineFooBroker(FooBroker):
    def __init__(self, profile, run_name: str, store_ctx: Any):
        super().__init__(symbol=profile.symbol, config=profile.config)
        self.profile = profile
        self.run_name = run_name
        self.store_ctx = store_ctx

    async def _http(self, method: str, path: str, payload=None):
        return self.profile.transport.respond(method, path, payload)


class FooProfile(ReferenceVenueProfile):
    plugin_name = "foo-offline-lab"
    account_id = "offline-account"
    symbol = "EURUSD"
    timeframe = "60"

    def create_broker(self, run_name: str, store_ctx: Any):
        return OfflineFooBroker(self, run_name, store_ctx)


def build_suite(*, mode: str, seed: int):
    smoke = Scenario(
        name="restart-adopts-working-order",
        profile_factory=FooProfile,
        seed=seed,
        steps=(
            Step("entry", values={"id": "L", "qty": 1.0, "limit": 1.1}),
            Step("sync"),
            Step("restart"),
            Step("entry", values={"id": "L", "qty": 1.0, "limit": 1.1}),
            Step("sync"),
            Step("expect", values={"open_orders": 1}),
        ),
    )
    return (smoke,)
```

Use a shared account-level `VenueState` for every broker instance created by a
profile. Separate `run_name` values represent isolated bot runs, while restart
creates a new plugin, engine, and SQLite connection over the same run and store.
Choose netting semantics when the venue exposes one aggregate symbol position.
For hedged venues, retain each venue position leg and provide the broker's real
`PositionPort` or an equivalent per-leg reconstruction in the profile.

### Replacing transports

- For HTTP, override the request dispatcher and return recorded response
  objects. Record method, path, parameters, and body so scenarios can assert the
  real plugin's request mapping.
- For WebSocket, feed the real message decoder from an in-memory queue. Model
  correlated ACK and PUSH independently: either may be delayed, duplicated,
  reordered, or intentionally absent.
- For a custom wire protocol, replace the lowest correlated request method and
  return real decoded protocol messages. The cTrader suite, for example, keeps
  the protobuf request and response types while replacing the socket.

Do not patch production discovery or add runtime instrumentation. Subprocess
fixtures can inject a temporary `.dist-info/entry_points.txt` on their isolated
Python path. The lab blocks direct socket connection attempts, and every child
must have an external timeout and its own temporary work directory. Never put
credentials, demo accounts, or real order endpoints in an offline profile.

### Invariants and plugin-specific checks

The common profile checks economic position agreement, client-order-id
uniqueness, terminal quantities, run ownership, fill idempotency, and physical
TP/SL coverage. Add venue-specific checks in `VenueProfile.check_invariants()`
and return human-readable violations after calling the base implementation.
Useful plugin checks include reduce-only sibling cancellation, activity-event
fingerprint stability, per-leg hedged ownership, and exact price/quantity-grid
rounding.

Keep deliberately broken control profiles beside the suite. A control scenario
sets `expected_violation` to the relevant invariant text; it passes only when
the oracle detects the injected defect.

### Running and reproducing

Run the short named corpus explicitly:

```bash
python -m pynecore.testing.broker_lab run broker_lab/suite.py --mode smoke
```

Run deterministic pairwise combinations separately:

```bash
python -m pynecore.testing.broker_lab run broker_lab/suite.py \
  --mode extended --seed 1337
```

From the PyneSys workspace root, the shipped suites have these explicit entry
points (none of them is part of the normal `pytest .../tests/` commands):

```bash
# PyneCore reference corpus
python -m pynecore.testing.broker_lab run pynecore/broker_lab/suite.py --mode smoke
python -m pynecore.testing.broker_lab run pynecore/broker_lab/suite.py --mode extended

# Real CLI lifecycle with a temporary offline plugin
python -m pynecore.testing.broker_lab run \
  pynecore/broker_lab/subprocess_suite.py --mode smoke

# Venue plugin profiles
python -m pynecore.testing.broker_lab run \
  plugins/pynecore-bybit/broker_lab/suite.py --mode smoke
python -m pynecore.testing.broker_lab run \
  plugins/pynecore-capitalcom/broker_lab/suite.py --mode smoke
python -m pynecore.testing.broker_lab run \
  plugins/pynecore-ctrader/broker_lab/suite.py --mode smoke
```

Replace `smoke` with `extended` and add `--seed N` for a plugin's pairwise
corpus. The subprocess suite intentionally has only lifecycle smoke scenarios.
Its temporary provider supplies a finite OHLCV stream and injects discovery via
disposable `.dist-info` metadata; it never changes production plugin discovery.

On failure, the command prints the seed, invariant, minimized step sequence,
and a `--scenario ... --seed ...` reproduction fragment. The same suite and
seed must generate the same scenarios and minimized reproduction. Minimized
steps describe the smallest known model-level trigger; inspect the fake
transport log and SQLite journal to determine whether the fault is in request
mapping, event translation, persistence, or engine interaction.

The lab should stay fast and deterministic, but it is not a replacement for a
small, tightly controlled final live smoke. Any newly discovered venue semantic
must first be documented, then added to the offline profile before relying on a
regression scenario for it.

## Combining Capabilities

A plugin can combine multiple capabilities via multiple inheritance.  The
`Config` dataclass is shared — it belongs to the plugin itself, not to any
specific capability.  The `[Config]` type parameter goes on the **first** parent
class — it doesn't matter which one, since both `ProviderPlugin` and `CLIPlugin`
propagate it:

```python
from dataclasses import dataclass
import click
import typer
from pynecore.core.plugin import ProviderPlugin, CLIPlugin, override


@dataclass
class FooConfig:
    """Foo provider"""

    api_key: str = ""
    """API key for authentication"""

    use_sandbox: bool = False
    """Use sandbox environment"""


# [FooConfig] on the first parent — either order works
class FooPlugin(ProviderPlugin[FooConfig], CLIPlugin):
    """Provider with CLI management commands."""

    Config = FooConfig

    # --- ProviderPlugin: data downloading ---
    # (timeframe converters and update_symbol_info omitted for brevity)

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        # self.config is typed as FooConfig (via Generic)
        ...

    @override
    def download_ohlcv(self, time_from, time_to, on_progress=None,
                       limit=None, with_extra=False):
        ...

    # --- CLIPlugin: subcommands ---

    @staticmethod
    def cli() -> typer.Typer:
        app = typer.Typer(help="Foo management commands")

        @app.command()
        def status():
            """Show connection status."""
            typer.echo("Connected")

        return app

    # --- CLIPlugin: parameter hooks ---

    @staticmethod
    def cli_params(command_name: str) -> list[click.Parameter]:
        if command_name == "run":
            return [click.Option(["--sandbox"], is_flag=True, default=False)]
        return []
```

This single plugin:
- Downloads data via `pyne data download foo`
- Adds `pyne foo status` subcommand
- Injects `--sandbox` into `pyne run`
- Gets a `workdir/config/plugins/foo.toml` with the `FooConfig` fields

The config TOML file is auto-generated on first run with all defaults commented
out.  Users uncomment and edit what they need:

```toml
# Foo provider

# API key for authentication
#api_key = ""

# Use sandbox environment
#use_sandbox = false
```

## Plugin Configuration

Any plugin type (`ProviderPlugin`, `CLIPlugin`, or a combination) can have a
`Config` dataclass.  Just set the `Config` class attribute and PyneCore handles
the rest.

The TOML file is:
- **Auto-generated** on first run with all defaults commented out
- **Self-healing** — new fields appear automatically, removed fields disappear
- **User-friendly** — docstrings become TOML comments, uncommented values survive regeneration
- **Cached** — `ensure_config()` returns the same instance on repeated calls

For ProviderPlugin, the config is automatically loaded and passed via
`self.config`.  For CLI-only plugins, load it manually:

```python
from pynecore.core.config import ensure_config
from pynecore.cli.app import app_state

config = ensure_config(FooConfig, app_state.config_dir / "plugins" / "foo.toml")
```

## Plugin Metadata

Plugin metadata comes from `pyproject.toml` via `importlib.metadata` — not from
class attributes:

```bash
pyne plugin list          # List all installed plugins
pyne plugin info ccxt     # Show details, config fields, capabilities
```

The `plugin_name` class attribute is optional and only used for display:

```python
class FooPlugin(ProviderPlugin[FooConfig], CLIPlugin):
    plugin_name = "Foo Service"  # shown in `pyne plugin list`
```

## Package Naming Convention

| Type      | Package name              | Example                |
|-----------|---------------------------|------------------------|
| Official  | `pynesys-pynecore-<name>` | `pynesys-pynecore-foo` |
| 3rd party | `pynecore-<name>`         | `pynecore-bar`         |

## Dependencies

For plugins with CLI capabilities, depend on the `cli` extra:

```toml
dependencies = ["pynesys-pynecore[cli]>=6.5"]
```

This ensures Typer and Click are available.  For provider-only plugins,
the base `pynesys-pynecore>=6.5` dependency is sufficient.
