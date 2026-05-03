<!--
---
weight: 1103
title: "Contributing"
description: "Guide for contributing to PyneCore"
icon: "handyman"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Development", "Community"]
tags: ["contributing", "guidelines", "development-setup", "code-style", "pull-requests", "workflow"]
---
-->

# Contributing to PyneCore

## Introduction

Thank you for your interest in contributing to PyneCore! This document provides guidance for developers who want to contribute to this open-source project that brings TradingView Pine Script compatibility to Python.

PyneCore is designed with a focus on high performance, minimal external dependencies, and an intuitive Pine Script-like development experience in Python. We welcome contributions that enhance this vision.

## Setting Up Your Development Environment

### Prerequisites

- **Python Version**: 3.11 or newer (as specified in the project's requirements)
- **Git**: For version control
- **Virtual Environment**: To isolate dependencies

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PyneSys/pynecore.git
   cd pynecore
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install development dependencies**:
   ```bash
   # For development, install all dependencies including development tools
   pip install -e ".[all,dev]"
   ```

4. **Verify your setup**:
   ```bash
   # Run the test suite to ensure everything is working
   python -m pytest
   ```

## Project Structure

PyneCore is organized as follows:

```
pynecore/
├── src/                  # Source code
│   └── pynecore/         # Main package
│       ├── core/         # Core functionality and runtime components
│       ├── lib/          # Pine Script compatible function library
│       ├── transformers/ # AST transformers for Pine Script syntax
│       ├── types/        # Type definitions and interfaces
│       ├── utils/        # Utility functions and helpers
│       ├── cli/          # Command-line interface
│       └── providers/    # Data providers and integrations
├── tests/                # Test suite
├── docs/                 # Component-specific documentation
└── scripts/              # Utility scripts
```

Understanding this structure is essential for effective contribution.

## Development Guidelines

### Code Style

PyneCore follows these coding standards:

1. **PEP 8 compliance**: Try to keep line length around 100 characters, with a hard limit of 120 characters as specified in the `pyproject.toml`
2. **Type hints**: Use for all function arguments and return values
3. **Docstrings**: Use PyCharm's default "Sphinx-style" reStructuredText format (`:param:`, `:return:`, etc.) for docstrings
4. **Naming conventions**:
   - Functions and variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
5. **Comments**: Add explanatory comments for complex logic

### Example Code Style

```python
def calculate_moving_average(values: Series[float], length: int) -> Series[float]:
    """
    Calculate a simple moving average of a series.

    :param values: Input series of values
    :param length: Window length for the moving average
    :return: New series containing the moving average values
    """
    if length <= 0:
        raise ValueError("Length must be positive")

    # Implementation logic
    result = values.rolling_sum(length) / length
    return result
```

### Clean Code Principles

1. **Single Responsibility**: Functions and classes should do one thing well
2. **DRY (Don't Repeat Yourself)**: Avoid code duplication
3. **KISS (Keep It Simple)**: Prefer simple solutions over complex ones
4. **Early Returns**: Return early to reduce nesting and improve readability

## Testing

PyneCore uses a comprehensive testing system where tests themselves are often Pyne code. This "dogfooding" approach ensures tests run in the same environment as real user code.

### Running Tests

```bash
# Run the entire test suite
python -m pytest

# Run specific test files
python -m pytest tests/path/to/test_file.py

# Run tests with coverage report
python -m pytest --cov=pynecore
```

### Writing Tests

Tests in PyneCore follow a unique pattern where test files are also valid Pyne code:

```python
"""
@pyne
"""
from pynecore.lib import script, ta, close, plot

@script.indicator(title="Test Indicator")
def main():
    my_value = ta.sma(close, 10)
    plot(my_value, "My SMA")

def __test_my_feature__(csv_reader, runner, dict_comparator, log):
    """ Test Description """
    with csv_reader('test_data.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(plot, candle.extra_fields)
            if i > 100:
                break
```

For more details, refer to the [Testing System Documentation](./testing-system.md).

## Working with Core Components

### AST Transformers

When modifying AST transformers in `pynecore/transformers/`:

1. Understand the transformation pipeline and execution order
2. Test with various edge cases
3. Ensure backward compatibility
4. Add comprehensive tests

### Library Functions

When working with library functions in `pynecore/lib/`:

1. Match Pine Script behavior when possible (for compatibility)
2. Document any differences from Pine Script
3. Optimize for performance
4. Add appropriate type hints
5. Include examples in docstrings

### Core Runtime

When modifying core runtime components in `pynecore/core/`:

1. Be extra cautious - these changes affect all Pyne code
2. Consider backward compatibility
3. Document performance implications
4. Add thorough tests

## Documentation

All code should be well-documented:

1. Every function should have a reStructuredText format docstring
2. Complex code should have inline comments
3. Major features should have documentation files

### Docstring Format Example

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.

    Detailed description of the function, if needed.

    :param param1: Description of param1
    :param param2: Description of param2
    :return: Description of return value
    :raises ExceptionType: When and why this exception is raised
    """
    # Function implementation
```

This is the PyCharm's default Sphinx-style docstring format, not the full reStructuredText syntax used for comprehensive documentation.

## Contribution Workflow

1. **Select or create an issue**: Start by finding an existing issue to work on or create a new one

2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**: Follow the coding standards and guidelines

4. **Add tests**: Ensure your changes are properly tested

5. **Run tests locally**:
   ```bash
   python -m pytest
   ```

6. **Commit your changes**:
   ```bash
   git commit -m "Description of changes"
   ```

7. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a pull request**: Provide a clear description of your changes

## Pull Request Guidelines

When submitting a pull request, please:

1. **Reference issues**: Link to any related issues
2. **Describe changes**: Provide a clear description of what you've changed
3. **Document impact**: Note any performance, compatibility, or API impacts
4. **Include tests**: Add appropriate tests for your changes
5. **Update documentation**: If applicable

## License

By contributing to PyneCore, you agree that your contributions will be licensed under the project's [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Additional Resources

- [Project Structure](../overview/project-structure.md)
- [Testing System Overview](./testing-system.md)
- [AST Module Documentation](https://docs.python.org/3/library/ast.html)
- [PyneCore Reference](/docs/reference/) - Complete function and type reference
- [TradingView Pine Script Documentation](https://www.tradingview.com/pine-script-docs/welcome/) - The original Pine Script language documentation

## Getting Help

If you have questions or need assistance, you can:

- Open an issue on GitHub
- Reach out to the maintainers

Thank you for contributing to PyneCore!