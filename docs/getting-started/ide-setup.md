<!--
---
weight: 204
title: "IDE Setup"
description: "Choose an editor for Pyne code and for PyneCore development. PyneIDE sets up a Pyne project by itself; other editors need a small type checker configuration."
icon: "code"
date: "2025-03-31"
lastmod: "2026-11-15"
draft: false
toc: true
categories: ["Getting Started"]
tags: ["ide", "setup", "pyneide", "vs-code", "pycharm", "pylance", "type-checking", "development-environment"]
---
-->

# IDE Setup

Which editor to use depends on what you are working on:

| You are...                           | Use                                       |
|--------------------------------------|-------------------------------------------|
| Writing Pyne code (your own scripts) | PyneIDE                                   |
| Using another editor for Pyne code   | PyCharm, or VS Code with a pyright config |
| Developing PyneCore itself           | PyCharm or VS Code                        |

## Writing Pyne Code: PyneIDE

PyneIDE is the Visual Studio Code extension for Pyne code and Pine Script, currently in
preview. It runs your scripts on an interactive chart, debugs them bar by bar, and checks
Pyne code with analysis that understands `Series` history, `na` and persistent state.

### Setup

1. Install **PyneIDE** (publisher: PyneSys) from the Visual Studio Code Marketplace.
2. Open a folder and run **PyneIDE: Initialize Pyne Project** from the Command Palette.
3. When prompted, let PyneIDE set up the Python environment. This is a one-time download
   of about 80 MB (`uv`, Python and PyneCore).

No manual configuration is needed. Initializing the project:

- creates a separate Python environment for the project in `.pyneide/venv`, with PyneCore
  and the debugger installed,
- writes a `pyrightconfig.json` for Pyne code and keeps it in sync with the interpreter
  PyneIDE runs scripts with,
- turns Pylance off for this workspace only (`python.languageServer: "None"`), so the type
  checker bundled with PyneIDE handles Python analysis there. Other workspaces keep
  Pylance. A value you already set in the workspace settings is left as it is.

See the PyneIDE README in the Marketplace for the full feature list and settings.

## Other Editors for Pyne Code

Pyne code is plain Python, so any editor works. General Python type checkers do not know
how PyneCore treats `Series` values, though, and need some help.

### Why Type Checkers Need Configuration

A `Series` value is used in two ways at once:

1. as a value in calculations, e.g. `(high + low) / 2`,
2. as a history you can index, e.g. `close[1]` for the previous bar's value.

Python typing has no intersection type, so this cannot be expressed directly. PyneCore's
stubs make `Series[T]` a transparent alias of `T`, which makes the value side type-correct.
The indexing side depends on the type checker, selected by a `TYPECHECKER` constant:

- **PyCharm**: `float`, `int` and `bool` are replaced by indexable stand-ins, so `x[1]`
  type-checks. No configuration is needed.
- **Pyright / Pylance** (VS Code and its forks): the builtins stay unchanged, and the
  configuration below switches off the checks that indexing and PyneCore's AST
  transformations would trigger.

### Pyright Configuration for VS Code

Without PyneIDE, create a `pyrightconfig.json` in the root of your project:

```json
{
   "reportIndexIssue": "none",
   "typeCheckingMode": "basic",
   "reportAssignmentType": "none",
   "reportRedeclaration": "none",
   "reportArgumentType": "none",
   "defineConstant": {
      "TYPECHECKER": "pyright"
   }
}
```

- `reportIndexIssue`: allows indexing a Series, e.g. `close[1]`
- `typeCheckingMode`: uses the relaxed `basic` mode
- `reportAssignmentType`: allows a Series to be treated as a value and as a history
- `reportRedeclaration`: allows the redeclarations that AST transformations rely on
- `reportArgumentType`: allows Series arguments where a plain value is expected
- `defineConstant`: selects the pyright variant of PyneCore's stubs

Some false positives remain even with this configuration.

## Developing PyneCore Itself

For work on PyneCore's own source, PyCharm and VS Code both work well. The repository
ships its own `pyrightconfig.json`, so Pylance and pyright pick up the right settings
when you open the repository root. See [Contributing](../development/contributing.md)
for the development environment, code style and tests.
