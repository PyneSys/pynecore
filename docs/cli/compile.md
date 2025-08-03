<!--
---
weight: 304
title: "Compiling Pine Scripts"
description: "Compiling Pine Script to Python using PyneSys API"
icon: "code"
date: "2025-08-03"
lastmod: "2025-08-03"
draft: false
toc: true
categories: ["Usage", "CLI", "Compilation"]
tags: ["compile", "pine-script", "python", "pynesys", "api", "command-line"]
---
-->

# Compiling Pine Scripts

The `compile` command allows you to convert Pine Script (.pine) files to Python (.py) files using the PyneSys API. This enables you to migrate your TradingView Pine Script strategies to PyneCore.

## Basic Usage

The basic syntax for compiling a Pine Script is:

```bash
pyne compile [SCRIPT] [OPTIONS]
```

Where:
- `SCRIPT`: Path to Pine Script file (.pine extension) or name of script in scripts directory (optional)
- `OPTIONS`: Additional options to customize the compilation

## Simple Example

```bash
# Compile a Pine Script file
pyne compile my_strategy.pine
```

This command will:
1. Look for `my_strategy.pine` in the current directory or `workdir/scripts/` directory
2. Compile it using the PyneSys API
3. Save the compiled Python code as `my_strategy.py` in the same location

## API Key Requirements

To use the compile command, you need a valid PyneSys API key. You can obtain one at [https://pynesys.io](https://pynesys.io).

The API key can be provided in several ways:

1. **Command line option**: Use the `--api-key` flag
2. **Environment variable**: Set `PYNESYS_API_KEY`
3. **Configuration file**: Store in `workdir/config/api.toml`

### Configuration File Setup

Create a file at `workdir/config/api.toml` with your API key:

```toml
[pynesys]
api_key = "your-api-key-here"
```

## Command Options

The `compile` command supports several options:

### Output Options

- `--output`, `-o`: Specify the output Python file path. Defaults to the same name with `.py` extension.

Example:
```bash
# Compile with custom output path
pyne compile my_strategy.pine --output ./compiled/my_strategy.py
```

### Compilation Options

- `--strict`, `-s`: Enable strict compilation mode with enhanced error checking
- `--force`, `-f`: Force recompilation even if output file is up-to-date
- `--api-key`, `-a`: PyneSys API key (overrides configuration file)

Example:
```bash
# Compile with strict mode and force recompilation
pyne compile my_strategy.pine --strict --force
```

### Usage Statistics

- `--usage`, `-u`: Print API usage statistics after compilation

Example:
```bash
# Show usage statistics
pyne compile my_strategy.pine --usage
```

## Pine Script Version Support

**Important**: Only Pine Script version 6 is supported. Version 4 and 5 scripts are not supported.

Make sure your Pine Script starts with:
```pine
//@version=6
```

## Usage Statistics

If you run the compile command without specifying a script, it will display your current API usage statistics and exit:

```bash
# Display usage statistics
pyne compile
```

## Working Directory Integration

The compile command integrates with PyneCore's working directory system:

- **Script lookup**: If you provide just a filename, it will be searched in the `workdir/scripts/` directory
- **Output location**: Compiled files are saved in the same directory as the source file
- **Configuration**: API configuration is read from `workdir/config/api.toml`

## Examples

### Basic Compilation

```bash
# Compile a Pine Script in the current directory
pyne compile my_strategy.pine
```

### Compilation with API Key

```bash
# Compile with API key provided via command line
pyne compile my_strategy.pine --api-key "your-api-key"
```

### Strict Mode Compilation

```bash
# Compile with enhanced error checking
pyne compile my_strategy.pine --strict
```

### Force Recompilation

```bash
# Force recompilation even if output is up-to-date
pyne compile my_strategy.pine --force
```

### Custom Output Path

```bash
# Compile to a specific output location
pyne compile my_strategy.pine --output ./strategies/compiled_strategy.py
```

### Show Usage Statistics

```bash
# Compile and show API usage statistics
pyne compile my_strategy.pine --usage
```

## Troubleshooting

### API Key Issues

```
Error: No API key provided
```

This error occurs when no API key is found. Make sure to:
- Set the `PYNESYS_API_KEY` environment variable, or
- Use the `--api-key` command line option, or
- Create a configuration file at `workdir/config/api.toml`

### Authentication Errors

```
Error: Invalid API key
```

This error occurs when the API key is invalid or expired. Check:
- Your API key is correct
- Your subscription is active
- You have remaining API credits

### Pine Script Version Errors

```
Error: Unsupported Pine Script version
```

This error occurs when trying to compile Pine Script v4 or v5. Only version 6 is supported.

### File Not Found

```
Script file 'my_strategy.pine' not found!
```

This error occurs when the Pine Script file cannot be found. Make sure:
- The file exists in the specified location
- If you provided just a filename, check that it exists in the `workdir/scripts/` directory
- The filename is spelled correctly (case sensitive)