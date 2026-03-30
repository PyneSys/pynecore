<!--
---
weight: 1104
title: "Plugin System"
description: "How to create plugins for PyneCore"
icon: "extension"
date: "2026-03-30"
lastmod: "2026-03-30"
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
├── ProviderPlugin   — Offline OHLCV data provider
├── CLIPlugin        — CLI subcommands and parameter hooks
├── ExtensionPlugin  — Hook-based script extension (planned)
└── LiveProviderPlugin — WebSocket/streaming data (planned)
```

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

Inject flags into existing commands (currently `run` is pluggable):

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
from dataclasses import dataclass
from pynecore.core.plugin import ProviderPlugin, override


@dataclass
class FooConfig:
    """Foo provider"""

    api_key: str = ""
    """API key for authentication"""

    use_sandbox: bool = False
    """Use sandbox environment"""


class FooProvider(ProviderPlugin[FooConfig]):
    Config = FooConfig

    @override
    def get_available_symbols(self) -> list[str]:
        ...

    @override
    def download(self, days_back, on_progress=None, extra_field_names=None):
        ...
```

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

    @override
    def get_available_symbols(self) -> list[str]:
        # self.config is typed as FooConfig (via Generic)
        ...

    @override
    def download(self, days_back, on_progress=None, extra_field_names=None):
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
