# Contributing to PyneCore

Bug reports and pull requests are welcome. These rules keep review fast for both sides.

## Bug reports

Open an issue with a minimal script that reproduces the problem, the data it runs on (or how to
get it), and what you expected instead. When the expected value comes from TradingView, say so.

- **One problem per issue.** Each issue is checked, fixed, released and closed on its own, and
  a fix in PyneCore ships separately from a fix in the PyneComp compiler. If you found several
  problems, open one issue for each. When they come from one investigation, you can also open
  an overview issue that links them.
- **Say where it happens:** the PyneCore runtime, the PyneComp compiler (Pine Script to Pyne
  code), the `pyne` CLI or a plugin. For a compiler problem, include the Pine Script source and
  the code PyneComp produced.

An issue that reports several problems is not closed for that reason, but we will ask you to
split it before we work on it.

## Pull requests

- **Base it on the current `main`.** Rebase before you open the PR and whenever `main` moves.
  A PR must merge into `main` without conflicts.
- **One fix per PR.** Unrelated changes go into separate PRs.
- **Describe only what the diff changes.** Do not list fixes that are already in `main` or that
  live in another PR.
- **Show the problem.** Give a script that fails or gives a wrong result on `main`, and add a
  test that fails without your change and passes with it.
- **Run the tests you touched:** `pytest tests/<path to the test>`.

## Conflicting pull requests are closed after 7 days

A daily job checks every open PR. When one no longer merges into `main`, it gets the
`merge-conflict` label and a comment. If the conflict is still there 7 days later, the PR is
closed. Rebase onto `main` and push to clear the label; a closed PR can be reopened after the
rebase.
