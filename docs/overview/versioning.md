<!--
---
weight: 109
title: "Versioning"
description: "Versioning policy for PyneCore"
icon: "history"
date: "2025-03-31"
lastmod: "2025-03-31"
draft: false
toc: true
categories: ["Overview", "Development"]
tags: ["versioning", "releases", "compatibility", "pine-script", "semver"]
---
-->

# Versioning Policy

`pynecore` uses a versioning system based on the Pine Script version supported by TradingView, extended with internal feature and patch levels:

```
<major>.<minor>.<patch>
```

## Breakdown:

- `major`: the supported Pine Script version (e.g., `6` means Pine v6)
- `minor`: new PyneCore-specific features that don’t affect Pine compatibility, but add capabilities
- `patch`: bug fixes and small improvements

## Examples:

- `6.0.0` – First stable release supporting Pine v6
- `6.0.4` – Bugfix release, still Pine v6
- `6.1.0` – Adds new Pine v6 features (e.g., support for `bar_index.new`)
- `7.0.0` – First version to support Pine v7

## Pre-release versions

When a new Pine version (e.g., v7) is released and still under integration/testing, pre-release versions will be published:

- `7.0.0a1` – Alpha release
- `7.0.0b1` – Beta release
- `7.0.0rc1` – Release candidate

These versions require explicit installation using the `--pre` flag in pip and are not installed by default.

This scheme ensures clarity for users and developers alike: stability is tied to the Pine version, while flexibility is maintained for PyneCore enhancements.

