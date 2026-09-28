<!--
---
weight: 109
title: "Versioning"
description: "Versioning policy for PyneCore"
icon: "history"
date: "2025-03-31"
lastmod: "2026-09-28"
draft: false
toc: true
categories: ["Overview", "Development"]
tags: ["versioning", "releases", "compatibility", "pine-script", "semver"]
---
-->

# Versioning Policy

`pynecore` uses a versioning system based on the Pine Script version whose API it follows, extended with PyneCore's own
release levels:

```
<major>.<minor>.<patch>
```

## Breakdown:

- `major`: the Pine Script version whose API PyneCore follows (e.g., `6` means Pine v6)
- `minor`: PyneCore's own major version, increased for breaking changes and other large changes
- `patch`: releases without breaking changes: bug fixes, improvements and smaller new features

## Examples:

- `6.0.0` – First stable release following the Pine v6 API
- `6.6.0` – Large change: live data and broker trading
- `6.10.3` – No breaking change: extended trading hours support
- `7.0.0` – First version to follow the Pine v7 API

## Pre-release versions

When a new Pine version (e.g., v7) is released and still under integration/testing, pre-release versions will be published:

- `7.0.0a1` – Alpha release
- `7.0.0b1` – Beta release
- `7.0.0rc1` – Release candidate

These versions require explicit installation using the `--pre` flag in pip and are not installed by default.

This scheme ties the major version to the Pine Script API, while the minor and patch levels follow PyneCore's own
development.
