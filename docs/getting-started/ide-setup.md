<!--
---
weight: 204
title: "IDE Setup"
description: "Setting up your IDE for PyneCore scripts"
icon: "code"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Getting Started", "Development Environment"]
tags: ["ide", "setup", "pycharm", "vs-code", "pylance", "type-checking", "development-environment"]
---
-->

# IDE Setup for PyneCore

This guide helps you set up your Integrated Development Environment (IDE) for the most effective experience when working with PyneCore in your own projects.

## Recommended IDEs

### PyCharm (Recommended)

PyCharm is the recommended IDE for PyneCore scripts due to its heuristic type checker which better accommodates PyneCore's dynamic behavior and AST transformations. PyCharm will generally work better "out of the box" when writing and running PyneCore scripts.

**Benefits for PyneCore users:**

- Heuristic type checking that understands dynamic Python behaviors
- Better handling of AST transformations that PyneCore uses internally
- More accurate code completion for Series operations and PyneCore functions
- **Perfect handling of Series types** - understands that they can be both indexed and used directly as values
- **No false positives** for PyneCore's dynamic features

#### Setup Instructions

1. Open your project in PyCharm
2. Ensure PyneCore is installed in your project's environment
3. No additional configuration is needed

### Visual Studio Code with Pylance

Visual Studio Code (or any fork of it like Cursor) with Pylance will work with PyneCore scripts, but you'll likely encounter many false positive errors due to VS Code's static type checking approach which cannot fully understand PyneCore's dynamic code transformation. **You will need additional configuration to suppress these irrelevant errors.**

#### Setup Instructions

1. Install the Python extension in VS Code
2. Ensure Pylance is enabled as your language server
3. Create a `pyrightconfig.json` file in your project root (see below)

## Required Type Checking Configuration for VS Code

When using PyneCore in your own projects with VS Code, **you will need to add a special configuration** to prevent overwhelming false error messages. This is not a bug in your code or in PyneCore - it's a fundamental limitation of static type checking with Python's dynamic features.

### The Series Type Challenge

The main challenge is that the **Series type** in PyneCore needs to function in two ways simultaneously:

1. As a container that can be indexed (e.g., `price[1]` to get the previous bar's value)
2. As a direct value that can be used in calculations (e.g., `(high + low) / 2`)

The different IDE type checkers handle this dual nature differently:

- **PyCharm**: Uses a heuristic approach that interprets `Union[T, SeriesType[T]]` as "this can be either type," allowing both behaviors without errors
- **Pylance**: Has a more strict implementation that treats Union types as exclusive, causing conflicts when the same variable is used both ways

PyneCore includes different type stub implementations optimized for each IDE, but you'll still need additional configuration for VS Code.

### Creating the PyRight Configuration File

Create a file named `pyrightconfig.json` in the root directory of your project with the following content:

```json
{
   "reportIndexIssue": "none",
   "typeCheckingMode": "basic",
   "reportUnknownMemberType": "none",
   "reportAssignmentType": "none",
   "reportRedeclaration": "none",
   "reportArgumentType": "none"
}
```

### Why You Need This Configuration

Without this configuration, VS Code will show many red squiggly lines and error messages in PyneCore scripts, even though your code is perfectly valid and will run correctly. These errors are triggered by several aspects of PyneCore:

1. The **Series type's dual nature** (container and direct value) confuses static type checking
2. **AST transformations** modify your code at import time in ways static analyzers can't predict
3. PyneCore's **dynamic property creation** and operations aren't visible to static analysis

Each setting in the configuration addresses a specific issue:

- `reportIndexIssue`: Disables errors when using Series as an indexable object (e.g., `close[1]`)
- `typeCheckingMode`: Relaxes type checking to accommodate PyneCore's dynamic behavior
- `reportUnknownMemberType`: Prevents errors with dynamically created properties
- `reportAssignmentType`: Avoids errors when a Series is treated as both a value and a container
- `reportRedeclaration`: Prevents errors from AST transformations that modify variable declarations
- `reportArgumentType`: Prevents errors with functions that accept Series arguments

### Type Stubs

PyneCore includes specialized type hint stubs for different IDEs:

- **PyCharm stubs**: Use a Union type that PyCharm interprets correctly for Series dual behaviors
- **Pylance stubs**: Implement a more complex approach, but still require configuration to work well

The system automatically selects the appropriate stubs based on the detected environment, but the fundamental limitations of static type checking still apply to Pylance.

## Recommended IDE Extensions

### PyCharm

- No additional extensions required for PyneCore functionality
- PyCharm Professional offers even better type inference capabilities

### VS Code

- Python extension (includes Pylance)
- Even with the configuration, expect to see some false positive errors

## Conclusion

**Key takeaways for PyneCore users:**

1. **PyCharm** provides a perfect experience with PyneCore's Series types and dynamic features due to its heuristic type checker

2. If using **VS Code**, you should add the `pyrightconfig.json` file to your project to reduce false error messages

Choose the IDE that best fits your workflow, but be aware of the limitations and necessary configurations when working with PyneCore's dynamic features.
