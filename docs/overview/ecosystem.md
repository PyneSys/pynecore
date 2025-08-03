<!--
---
weight: 102
title: "Pyne Ecosystem"
description: "Overview of the complete Pyne ecosystem and how its components work together"
icon: "lan"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Overview", "Ecosystem"]
tags: ["ecosystem", "pynecomp", "pynecore", "services", "business-model", "community"]
---
-->

# Pyne Ecosystem

The Pyne ecosystem consists of multiple interconnected components that together form a complete solution for TradingView
Pine Script compatibility in Python. This page provides an overview of these components and how they work together.

## Core Components

The Pyne ecosystem is built around two main components:

### PyneCore (Open Source)

PyneCore is the foundation of the ecosystem - an open-source Python implementation of a Pine Script-like environment. It
provides:

- A Pine Script compatible runtime in Python
- AST transformations that enable Pine Script-like syntax and features
- A complete library of technical indicators and functions, comparable to Pine Script's library
- Persistent variables, which store their values between runs (every candle)
- Series data structures, which work the same way as Pine Script's series
- NA class, which works the same way as Pine Script's NA
- Strategy backtesting capabilities, compatible with Pine Script's strategy tester

PyneCore allows you to write Pine Script-like code directly in Python, leveraging Python's ecosystem while maintaining
the advantages of Pine Script's execution model.

Learn more about PyneCore in the [What is PyneCore](/docs/overview/what-is-pynecore/) page.

### PyneComp - Pine Script to PyneCore Compiler/Transpiler (SaaS Service)

PyneComp is a compiler service that translates existing Pine Script code into PyneCore-compatible Python code. It
offers:

- Clean, readable Python code generation (PyneCore)
- Strict mode with full scope isolation (it is not needed most of the time)
- 100% Pine Script compatibility

This service is available through:

- The PyneSys API
- The [pynesys.io](https://pynesys.io) web interface
- Direct integration with PyneCore CLI (if you have API key)

PyneComp enables a smooth migration path from TradingView Pine Script to Python with minimal effort. PyneCore has all
the tools to run the compiled python code.

## Planned Services

The following services are planned and/or already being developed:

### MetaTrader 4/5 Compiler (Transpiler)

We are planning a compiler which can convert Pine Scripts to MetaTrader 4/5 expert advisors. We have the knowledge and
experience to do this.

### Strategy Leaderboard

An online automatic strategy ranking system where:

- Users can upload their (own) trading strategies
- The system runs backtests regularly across multiple markets and timeframes
- Strategies are ranked based on performance
- Users can select strategies based on recent performance indicators
- Weekly performance metrics help identify situational strengths

### Cloud Robot (PyneBot)

A cloud-based execution environment for trading algorithms that:

- Runs your PyneCore scripts in the cloud
- Connects to various brokers and exchanges
- Provides 24/7 monitoring and execution
- Offers detailed performance analytics

### Strategy Marketplace

A platform for buying and selling trading strategies:

- Vetted, high-quality strategies
- Performance statistics and validation
- Secure licensing and delivery
- Revenue sharing with strategy creators
- Developers can sell their strategies on the marketplace with or without source code

## Use Cases

The Pyne ecosystem serves multiple use cases:

### For TradingView Users

- Migrate existing Pine Script strategies to Python
- Overcome TradingView limitations (data access, execution)
- Enhance strategies with Python libraries (ML, optimization)

### For Python Developers

- Use familiar Python syntax for trading strategies
- Access Pine Script's powerful technical analysis functions
- Leverage Python's rich ecosystem for trading

### For Institutional Traders

- Standardize trading algorithm codebase
- Improve backtesting capabilities
- Integrate with existing Python infrastructure
- Optimize execution performance

## Business Model

The Pyne ecosystem combines open-source and commercial components:

- **PyneCore**: Free and open-source (Apache 2.0 license)
- **PyneComp**: Subscription-based service with multiple tiers
- **Cloud Robot**: Subscription with usage-based pricing
- **Strategy Marketplace**: Commission-based platform

## Community and Support

Join our community to get help and share your experiences:

- [Discord Server](https://discord.com/invite/jegnhtq6gy)
- [Reddit](https://www.reddit.com/r/PyneSys)
- [GitHub Discussions](https://github.com/PyneSys/pynecore/discussions)
