# #48 scenario (c) — per-strategy attribution on a netted account (DESIGN, operator decision pending)

The measured problem (matrix probe c, recon 2026-08-25): DNSE nets per symbol;
`run_id` keys on `strategy_id`, so two long strategies plus a short hedge open
the SAME account with no error, no lease, no attribution. Each engine sees the
others' fills only as unexplained drift — which, since the #48 detector, warns
but cannot attribute. Nothing in core or plugin owns the question "whose lot
was that?".

## Option 1 — venue-level isolation: one sub-account per strategy
Each strategy gets its own DNSE sub-account; the venue itself keeps the books
separate, attribution is free, and the netting problem disappears.
- Cost: operator-side account admin; needs venue verification that sub-accounts
  net independently and share margin the way we assume (UNMEASURED).
- Limits: does not scale to many strategies; the operator's own manual trading
  still needs its own compartment.

## Option 2 — engine-level net-share ledger (post-#36)
Every fill journaled per run (`orders` table — the #36 store seam); a shared
per-account ledger derives each run's owned signed size; venue net minus the
sum of owned sizes = the "unowned residual" (the operator's manual book). The
startup clamp (`_durable_owned_signed_size`) already implements the read side
of exactly this — it just has nothing to read for DNSE until #36 lands.
- Cost: blocked on #36; needs disciplined journal writes on every fill path;
  residual attribution is only as good as fill capture.
- Wins: attribution AND the drift detector can then separate "another strategy"
  from "external interference" instead of one blended warning.

## Option 3 — refuse-multi-writer guard (cheap, immediate)
A netted-account lease in `BrokerStore` (the derivatives twin of
`claim_spot_asset`, storage.py:2338): the second engine opening the same
`(account, symbol)` FAILS LOUDLY at `open_run` unless the operator passes an
explicit `--share-account` override. Prevents the hazard instead of solving
attribution; composes with either option above later.
- Cost: ~30 lines in storage + a config/CLI flag; no venue dependency.
- Limits: blocks the 2-long + 1-hedge fleet the operator actually wants —
  usable only as a default-with-override, never a hard wall.

## Recommendation (not a decision)
Option 3 now (default-on lease with explicit override), Option 2 when #36
lands (it reuses the clamp's machinery), Option 1 where the venue makes it
cheap. The three compose; none is exclusive.

**Operator decision needed before any code**: which option(s), and whether the
lease default is refuse-with-override or warn-and-continue.
