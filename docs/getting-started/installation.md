<!--
---
weight: 201
title: "Installing PyneCore"
description: "Step-by-step guide to installing PyneCore on different platforms"
icon: "download"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Getting Started"]
tags: ["installation", "setup", "pip", "requirements", "workdir", "verification"]
---
-->

# Installing PyneCore

This guide will walk you through the process of installing PyneCore on your system.

## System Requirements

Before installing PyneCore, ensure your system meets the following requirements:

- **Python Version**: 3.11 or newer (as specified in the project's requirements)
- **Operating System**: Windows, macOS, or Linux
- **Required Dependencies**: The core system has minimal dependencies, but some features require additional packages

### Windows-specific Requirements

Windows users need to be aware that Python's `zoneinfo` module requires timezone data that is not included in Windows by default. This is automatically handled when installing with the `[cli]` or `[all]` options, which include the `tzdata` package. If you encounter timezone-related errors with the basic installation, you'll need to install `tzdata` manually.

## Installation Methods

### Method 1: Using pip (Recommended)

The simplest way to install PyneCore is using pip:

```bash
# Install without any optional dependencies
pip install pynesys-pynecore
```

This installs the library with no dependencies at all. It is enough to run Pyne code from Python (for example a compiled script with `python script.py data.csv`), but the `pyne` command needs the `[cli]` extra.

To install with a user-friendly CLI (recommended):

```bash
pip install "pynesys-pynecore[cli]"
```

To install with additional features, you can specify optional dependencies:

```bash
# Install with the built-in CCXT data provider
pip install "pynesys-pynecore[providers]"

# Install only the CCXT library for the built-in CCXT data provider
pip install "pynesys-pynecore[ccxt]"

# Install for development
pip install "pynesys-pynecore[dev]"
```

Other data sources and brokers are separate plugins.

These extras do not include the CLI. Combine them with `cli` (for example `"pynesys-pynecore[cli,providers]"`) or use `[all]`.

To install with all features without development dependencies:

```bash
pip install "pynesys-pynecore[all]"
```

If you want to install all features with development dependencies:

```bash
pip install "pynesys-pynecore[all,dev]"
```

### Method 2: From Source (for developers)

For developers who want the latest development version or plan to contribute:

```bash
# Clone the repository
git clone https://github.com/PyneSys/pynecore.git

# Change to the PyneCore directory
cd pynecore

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"
```

## Setting Up the Working Directory

PyneCore uses a "workdir" directory structure that contains your scripts, data, and configuration. When you run PyneCore, it automatically searches the current and parent directories for a `workdir` folder (or a directory containing a `.pyne` marker file). You can also set it explicitly with the `--workdir` (`-w`) option or the `PYNE_WORK_DIR` environment variable.

The working directory structure is organized as follows:

```
workdir/
├── .pyne        # Workdir marker
├── scripts/     # Your Pyne code (lib/ for your own libraries, demo.py)
├── data/        # OHLCV data files with their .toml symbol info (demo.ohlcv)
├── output/      # Output files (plots, strategy results, trades), logs/
├── cache/       # Runtime state (e.g. auth sessions)
└── config/      # api.toml, plugins/*.toml
```

If no workdir exists, the PyneCore CLI offers to create `./workdir` with the full structure, including a demo script and demo data.

## Verifying Installation

To verify that PyneCore is installed correctly, run the following command:

```bash
pyne -h
```

You should see the PyneCore logo and help information for the CLI.

### Creating a Test Script

Create a simple test script to verify everything is working:

```python
# workdir/scripts/test.py
"""
@pyne
"""
from pynecore.lib import script

@script.indicator("Test")
def main():
    print("PyneCore is working!")
```

## Downloading Sample Data

PyneCore includes a data command for downloading historical OHLCV data from various providers. You'll need to install the CLI and the provider dependencies first:

```bash
pip install "pynesys-pynecore[all]"
```

To download sample data:

```bash
# List installed providers (or run `pyne data download` without arguments to pick one interactively)
pyne plugin list

# Example: Download Bybit data (if supported)
pyne data download ccxt:BYBIT:BTC/USDT:USDT@1D
```
By default, PyneCore will download 1 year of data. You can change this by using the `--from` flag. It can specify a concrete date or if it is just a number, you can specify how many days back you want to download:

```bash
# Download 100 days of data
pyne data download ccxt:BYBIT:BTC/USDT:USDT@1D --from 100

# Download data from a specific date
pyne data download ccxt:BYBIT:BTC/USDT:USDT@1D --from 2024-01-01
```

You can also convert your existing CSV or JSON containing OHLCV data to the PyneCore's `.ohlcv` format:

```bash
pyne data convert-from path/to/your/data.csv --symbol "BTCUSDT"
```

## Running Your First Script

After installing PyneCore and setting up data, you can run a simple script:

```bash
pyne run test your-downloaded-data
```

Bare names are looked up in `workdir/scripts` and `workdir/data`, and the extensions are optional.

## Next Steps

Now that you have PyneCore installed, you're ready to [create your first script](./first-script.md).
