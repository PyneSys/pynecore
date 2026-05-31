<!--
---
weight: 304
title: "Symbol Browser TUI"
description: "Interactive symbol browser for pyne data download"
icon: "list"
date: "2026-05-17"
lastmod: "2026-05-17"
draft: false
toc: true
categories: ["Usage", "CLI", "Data Handling"]
tags: ["cli", "tui", "data", "download", "symbols"]
---
-->

# Symbol Browser TUI

The Symbol Browser is the interactive two-pane TUI that `pyne data download` drops into when no `--symbol` is given on the command line. It lets you scroll the provider's full symbol list with live symbol-info on the side, then start a download through an inline wizard — without leaving the terminal.

## When it opens

The browser opens automatically when:

- you invoke a provider without a symbol — a bare provider name or, for a multi-broker provider, a broker-only provider string — **and**
- stdin is a real terminal (TTY).

If stdin is piped or redirected, the command errors out with `Symbol is required (or use --list-symbols for non-interactive listing).` — the TUI is interactive-only.

```bash
# Open the browser for Capital.com (no symbol given)
pyne data download capitalcom

# Same, but pre-fill the wizard's timeframe field with 15
pyne data download capitalcom --timeframe 15

# Multi-broker provider: name the broker to scope the symbol list
pyne data download ccxt:BINANCE
```

`--timeframe`, `--from` and `--to` from the command line are not enforced inside the TUI — they become the **defaults** of the wizard's matching fields and can be changed before submitting.

## Layout

```
+-------------------------+----------------------------------+
| Symbols (list pane)     | Symbol info (details pane)       |
| > EURUSD                | Description: EUR/USD             |
|   GBPUSD                | Type: forex                      |
|   USDJPY                | Tick size: 0.00001               |
|   ...                   | Opening hours: ...               |
+-------------------------+----------------------------------+
| Wizard / progress strip (only in wizard or downloading)    |
+------------------------------------------------------------+
| Footer: contextual key hints                               |
+------------------------------------------------------------+
```

- **Left pane** — scrollable list of every symbol the provider returned. The current cursor row is highlighted; the visible window auto-scrolls to keep the cursor in view.
- **Right pane** — symbol info (`SymInfo`) for the row under the cursor: description, asset type, tick size, opening hours, etc. Fetched on demand on a single background worker, debounced by ~150 ms so fast scrolling does not flood the provider with requests, and cached in memory with an LRU policy.
- **Wizard / progress strip** — only present while you are configuring or running a download. In browse mode this region is empty.
- **Footer** — one line of context-dependent key hints. After a download completes (or fails), the footer shows the result line in green / red for a few seconds before fading back to the help text.

## Modes

The browser has three modes, and the active mode determines which keys do what:

| Mode          | When                                                                     |
|---------------|--------------------------------------------------------------------------|
| `browse`      | Default — moving the cursor through the symbol list.                     |
| `wizard`      | After pressing **Enter** on a symbol — configuring timeframe and dates.  |
| `downloading` | After submitting the wizard — a progress bar is running.                 |

A separate `filter_active` flag layers on top of `browse` while you are typing in the `/` search box.

## Browse mode — key bindings

| Key             | Action                                              |
|-----------------|-----------------------------------------------------|
| `Up` / `Down`   | Move cursor one row.                                |
| `PgUp` / `PgDn` | Move cursor 10 rows.                                |
| `Home` / `End`  | Jump to first / last symbol.                        |
| `Enter`         | Open the download wizard for the highlighted symbol.|
| `/`             | Start an incremental filter on the symbol list.     |
| `q` / `Esc`     | Quit the browser.                                   |

## Filter (`/` search)

Pressing `/` activates an incremental, case-insensitive substring filter on the symbol list. Anything you type narrows the visible list immediately. The filter has its own key handling, designed to avoid the "feels like the keypress was swallowed" pattern:

| Key             | Action while filtering                                                                          |
|-----------------|-------------------------------------------------------------------------------------------------|
| Printable chars | Append to the filter text and re-filter the list.                                               |
| `Backspace`     | Delete the last filter character; on empty buffer, exit the filter.                             |
| `Esc`           | Exit the filter **and** clear the filter text.                                                  |
| `Up` / `Down`   | Exit the filter **and** move the cursor. Filter text is kept (list stays narrowed).             |
| `PgUp` / `PgDn` | Exit the filter **and** move the cursor by 10. Filter text is kept.                             |
| `Enter`         | Exit the filter **and** open the wizard for the currently highlighted symbol — in one step.     |

The intent: once you have typed enough to pick out the row you want, a single `Enter` (or arrow-key + `Enter`) takes you straight into the download wizard. There is no separate "commit the filter first, then press Enter again" step.

## Wizard mode

`Enter` on a symbol opens the inline wizard below the panes. Fields shown:

| Field        | Purpose                                                                                                  |
|--------------|----------------------------------------------------------------------------------------------------------|
| `Timeframe`  | Bar size in TradingView format: `1`, `5`, `15`, `30`, `60`, `240`, `1D`, `1W`, `1M`, or `Custom...`.     |
| `From`       | Start date — `continue`, a preset days-back (`1`, `7`, `30`, `90`, `180`, `365`), or `Custom...`.        |
| `To`         | End date — `now` or `Custom...`.                                                                         |
| `Truncate`   | (Only shown if the target OHLCV file already exists.) `Yes` wipes the file before downloading.           |
| `[Download]` | Submit button — `Enter` here dispatches the download.                                                    |

### Field interaction

| Key                  | Action                                                                                  |
|----------------------|-----------------------------------------------------------------------------------------|
| `Tab` / `Left` / `Right` | Move focus to the next / previous field.                                            |
| `Up` / `Down`        | On a focused field: cycle the dropdown options.                                         |
| `Enter`              | Open the dropdown if closed; confirm the selection if open; on `Custom...` enter text mode; on the submit button start the download. |
| `Esc`                | Cancel the wizard and return to browse mode.                                            |

In `Custom...` text mode, type the value freehand:

- **Custom timeframe** — anything that `validate_timeframe()` accepts, e.g. `3`, `90`, `1D`.
- **Custom dates** — `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`. The `From` field also accepts the literal `continue`.

`Enter` commits the text, `Esc` reverts to the previous option.

### Smart defaults

The wizard adjusts itself based on whether the target file already exists for `<symbol, timeframe>`:

- **`Truncate` field** appears only when the target OHLCV file is already on disk. If you change the timeframe and the new target does not exist, the field is hidden again automatically.
- **`From` smart default** — when no `--from` was passed on the command line, `From` is initialised to `continue` if the file exists, otherwise to `365` days back. (An explicit `--from` from the CLI always wins.)

### Validation

The wizard validates inputs at submit time, not while typing. Invalid values keep you in the wizard with a red error line; the offending field re-takes focus. Typical errors: malformed date, unknown timeframe, end date earlier than start date, `continue` used on `To`.

## Download mode (progress strip)

Once you submit the wizard, the wizard strip is replaced by a single-line progress bar:

```
+-- Downloading EURUSD 15 --------------------------------------+
| EURUSD 15 [#####-----] 47%  0:00:08 / 0:00:09                 |
+---------------------------------------------------------------+
```

- **Label** — `<symbol> <timeframe>` from the wizard.
- **Bar / percent** — fraction of the requested **time range** that has been processed so far (not bytes, not bar count). The provider reports a timestamp on each chunk callback; the strip maps that onto the configured `From..To` window.
- **Elapsed** — real wall-clock seconds since the download was dispatched.
- **ETA** — projected remaining wall-clock time, computed as `elapsed_real * (total - completed) / completed`. It stays at `-:--:--` until the first chunk arrives, then settles down as the average rate stabilises.

The browser ignores keypresses during a download (Ctrl-C still escapes through the terminal). When the download finishes, the strip disappears and a one-line status banner takes over the footer for a few seconds:

- `[OK] downloaded <symbol> <timeframe>` (green) — success.
- `[ERR] <ExceptionType>: <message>` (red) — failure. The browser stays open so you can change parameters and retry.

## Caching and background fetching

The right pane uses a small in-memory LRU cache keyed by symbol. The first time the cursor lands on a symbol, the symbol-info fetch is queued on a single background worker thread (one in flight at a time, to keep provider concurrency predictable). Subsequent visits to that symbol are served from the cache.

A `~150 ms` debounce on the cursor protects the provider during fast scrolling: only symbols you actually stop on for that long trigger a fetch. Errors during fetch are cached too — the right pane shows the error message and we do not retry the same symbol automatically.

## Provider-specific notes

### Capital.com — weekend / market-closed gaps

When the `From..To` window runs past the last available bar (typically into a weekend on FX), the Capital.com `/prices` endpoint returns HTTP 404 with `error.prices.not-found` rather than an empty `prices` list. The browser treats this identically to "no more data": the download stops cleanly, everything written so far is kept, and the footer shows the normal `[OK] downloaded ...` line.

### CCXT — symbol list scope

CCXT supports 100+ exchanges; the symbol list is per-exchange. Name the broker in a broker-only provider string when opening the browser so the right list is fetched:

```bash
pyne data download ccxt:BINANCE
```

## Troubleshooting

- **TUI does not open, error about `Symbol is required`** — stdin is not a TTY (you are piping or redirecting). Add `--symbol` on the command line, or run the command interactively.
- **Right pane stuck on "Loading..."** — the provider is slow or the symbol has no info available. Move the cursor away and back to retry; cached errors are surfaced as such in the panel.
- **Progress bar shows `0:00:00` / `-:--:--`** — expected during the first second of the download, before the first chunk arrives. If it stays there longer, the provider may be slow on the first response; the bar updates as soon as a chunk reports back.
- **`Truncate` field missing** — the target OHLCV file does not exist yet, so there is nothing to truncate. Once the first download is on disk, the field will appear on the next wizard entry.
