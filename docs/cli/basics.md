<!--
---
weight: 301
title: "Basic Usage"
description: "Basic usage of the PyneCore Command Line Interface"
icon: "terminal"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Usage", "CLI"]
tags: ["cli", "command-line", "basics", "workdir", "commands"]
---
-->

# CLI Basic Usage

This page covers the basic usage of the PyneCore Command Line Interface (CLI), including command structure, and working directory concepts.

For CLI to work, you need to install PyneCore with the `cli` optional dependency:

```bash
pip install "pynesys-pynecore[cli]"
```

## Command Structure

The PyneCore CLI follows a consistent command structure:

```
pyne [command] [subcommand] [options] [arguments]
```

Where:
- `command`: The main command (e.g., `run`, `data`)
- `subcommand`: A specific action for the command (e.g., `data download`)
- `options`: Optional parameters that modify the command behavior (start with `-` or `--`)
- `arguments`: Required parameters for the command

## Common CLI Options

The PyneCore CLI supports the following global options that can be used with any command:

- `--workdir`, `-w`: Specify a custom working directory instead of using the auto-detected one. This overrides the automatic working directory detection.

  ```bash
  pyne --workdir /path/to/custom/workdir run my_script.py my_data.ohlcv
  ```

- `--help`, `-h`: Show help message and exit. This works with any command or subcommand.

  ```bash
  pyne --help
  pyne run --help
  ```

- `--install-completion`: Install shell completion for the current shell. This makes command-line completion work for PyneCore commands.

  ```bash
  pyne --install-completion
  ```

- `--show-completion`: Show completion script for the current shell, which you can copy or customize for installation.

  ```bash
  pyne --show-completion
  ```

You can also set the working directory using the `PYNE_WORK_DIR` environment variable:

```bash
PYNE_WORK_DIR=/path/to/custom/workdir pyne run my_script.py my_data.ohlcv
```

## Getting Help

You can get help on any command by adding the `-h` or `--help` flag:

```bash
# Get general help
pyne --help

# Get help on a specific command
pyne run --help

# Get help on a subcommand
pyne data download --help
```

## The Working Directory Concept

The PyneCore CLI uses a **working directory** (workdir) structure to organize files. This provides a consistent location for your scripts, data, and output files.

You can read more about the working directory concept in the [Working Directory](../overview/working-directory.md) page.

### Path Handling

When specifying file paths in CLI commands:

- **Full paths**: If you provide a full path, the system uses it directly
- **Relative paths**: If you provide a relative path, it's resolved relative to the current directory
- **Filenames only**: If you provide just a filename:
  - For scripts: It looks in `workdir/scripts/`
  - For data files: It looks in `workdir/data/`
  - For output files: It saves in `workdir/output/`

Example:
```bash
# Using just filenames (will use workdir)
pyne run my_script.py my_data.ohlcv

# Same as:
pyne run ./workdir/scripts/my_script.py ./workdir/data/my_data.ohlcv
```

## Command Output

PyneCore CLI uses rich formatting to display:

- Progress bars for long-running operations
- Colorized output for better readability
- Spinners for operations with unknown duration
- Time estimates for completion

Example output when running a script:
<pre>
  ____
 |  _ \ _   _ _ __   ___
 | |_) | | | | '_ \ / _ \
 |  __/| |_| | | | |  __/
 |_|    \__, |_| |_|\___|
        |___/

✓ Running script... [██████████████████████████████] 2023-01-01 00:00:00 / 0:01:45
</pre>

## Environment Variables

The PyneCore CLI behavior can be modified using environment variables:

- `PYNE_WORK_DIR`: Set the working directory path
- `PYNE_NO_LOGO`: Set to any value to disable the logo display
- `PYNE_QUIET`: Set to any value for quieter output (disables the logo and reduces verbosity)
- `NO_COLOR`: Set to any value to disable colored output

Example:
```bash
# Run without displaying the logo
PYNE_NO_LOGO=1 pyne run my_script.py my_data.ohlcv

# Specify a custom working directory
PYNE_WORK_DIR=/path/to/my/workdir pyne run my_script.py my_data.ohlcv
```

## Available Commands

The PyneCore CLI provides the following main commands:

- `run`: Run a PyneCore script (.py or .pine)
- `compile`: Compile Pine Script to Python using PyneSys API
- `data`: OHLCV related commands

## Next Steps

Now that you understand the basic concepts, you can learn about specific commands:

- [Running Scripts](run.md): How to run PyneCore scripts
- [Compiling Pine Scripts](compile.md): How to compile Pine Scripts to Python
- [Data Management](data.md): Data download and conversion commands