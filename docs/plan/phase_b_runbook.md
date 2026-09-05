# Phase B runbook — loop-resumable execution prompt (2026-09-05)

**Purpose:** drive Phase B of `docs/plan/dnse_v2_fix_plan.md` to completion
across MULTIPLE sessions fired by `/loop` (~5 h cadence, resuming after usage
limits). Each firing is a FRESH context: this file is the memory. Read it
fully, detect state, continue — never restart finished work.

**Operator constraints (absolute, every firing):**
- OFFLINE ONLY: no live venue writes, no `--broker`, no L0 — read-only venue
  calls allowed only if a card's panel demands a measurement.
- Every `gh` call carries `--repo rubycell/pynecore` (fork default = UPSTREAM).
- Every card comment passes `python3 ~/.claude/skills/backlog-startcard/scrub_check.py <file> --strict --repo <repo>` first.
- Never `rm` (backup/deleteable), never `pyne run --from`, never echo secrets.
- The FULL pipeline per card (backlog-startcard skill): red-first baseline →
  false-positive challenge → card body/Ready/active → SOLUTIONS FOR REVIEW →
  3-lens panel (general-purpose agents: correctness / operational / maintainability)
  → adjudicate (leader verifies claims in code) → implement → suites → decision
  comment → commit skill closeout (In review). No waivers.

## Step 0 — STATE DETECTION (every firing, before anything)

1. `git status -sb && git log --oneline -3`
2. `python3 ~/.claude/skills/backlog-startcard/backlog_index.py get-active`
3. `gh issue view <each card below> --repo rubycell/pynecore --json title,body,comments`
   — count SOLUTIONS/review/adjudication/DECISION comments to locate the step.
4. Uncommitted tracked changes? Run `pytest plugins/dnse/tests/ -q`:
   - green + coherent with the active card → continue that card mid-step;
   - broken → park the diff in `backup/` (zip), reset only YOUR uncommitted
     work, re-derive from the card thread.
5. A card is DONE when: DECISION comment posted, commit pushed, card In
   review, active task cleared. Continue to the NEXT card.
6. Panel died on a 429/spend-limit? Re-spawn ONLY the lenses whose review
   comment is missing from the thread (check the thread, not memory).

**Sequence:** B1 → B2 → B3 → (small) #69 → #70. When ALL are In review:
post a completion digest on #53, update the memory queue file, STOP THE LOOP
(if running under /loop dynamic mode, end it; otherwise tell the operator the
loop can be cancelled).

## Card B1 — recovery verdict ladder (+ #60)

- **Card:** create fresh ("dnse Phase B1: recovery verdict ladder over the
  #36 journal — two classes, #51 re-adoption, run-label strand report");
  `closes: [60]` candidate — confirm at commit. Read FIRST: #60's body, the
  #59 item-2 review comment (5420751110), plan doc item 2 + fork resolutions.
- **Scope:** startup recovery POLICY over #36's rows: (a) journalled ids whose
  engine envelope died → adopt/re-point, NEVER blind-cancel; (b) ACK-window
  rows (`disposition_unknown`, phase=sent) → resolve by read; unresolvable →
  DESIGNED HALT + operator list (BrokerManualInterventionError); (c) #51
  inversion: a conditional recovery cannot cancel gets RE-ADOPTED, never
  retire-and-replace; (d) #60: rows under a DIFFERENT run label are reported
  LOUDLY as stranded (never silently invisible, never adopted).
- **Test plan (`plugins/dnse/tests/test_recovery_ladder.py`), red-first:**
  R1 journalled-id crash → re-owned + cancellable (extends #36's B2 to the
  ladder classes); R2 disposition_unknown row + venue detail resolves →
  correct terminal/adoption; R3 disposition_unknown + unresolvable →
  designed halt naming the ids; R4 different-run-label rows → loud stranded
  report, zero adoption; R5 control: clean-shutdown store → no ladder action;
  R6 control: FOREIGN book rows untouched (re-pin through the ladder).
- **Hard questions for the panel:** does the ladder double-act with the
  engine's `store_ctx.replay()`; is halt-on-unresolvable safe at startup
  BEFORE any position exists; #60's correct semantics (report vs adopt — the
  operator shares the account).

## Card B2 — item-4 clamped-snapshot chokepoint

- **Card:** create fresh ("engine/dnse Phase B2: clamped-snapshot chokepoint —
  raw-vs-owned split for the four raw-net consumers"). Read FIRST: the #59
  item-4 review (5420844968) for the four consumers + probes; plan doc item 4
  + fork resolution (write-side split is NOT this card).
- **Scope:** one chokepoint producing the owned/clamped account snapshot,
  each engine consumer explicitly labeled raw-vs-owned; halt/settle proofs
  KEEP RAW; `baseline_established == False` → UNKNOWN, never 0. Engine-side
  code — smallest honest diff, upstream-rebase tax weighed by the panel.
- **Test plan (core-side, `tests/t00_pynecore/core/` conventions), red-first:**
  one anchor per consumer (adopt-size replay, periodic shrink-to-zero,
  partial-bracket parent snapshot, halt/settle proofs = RAW); plus
  no-baseline→UNKNOWN; plus an operator-position-present scenario (netting
  account: owned ≤ raw bound, never determined).
- **Hard questions:** which consumers are LOAD-BEARING vs the r2 enumeration
  (verify in today's sync_engine, line numbers moved); does the clamp change
  any #48 divergence-matrix expectation (run that suite).

## Card B3 — item-3 residue detector (tracker demoted)

- **Card:** create fresh ("dnse Phase B3: external-cancel residue detector —
  terminal transitions primary, history-only CANCELLED authority").
  Read FIRST: the #59 item-3 review (5420854335), plan doc item 3; note the
  `rows|None` first step ALREADY landed (#62/#54) — verify, don't redo.
- **Scope:** the narrow residue detector: an id that vanished from every book
  WITHOUT a terminal transition ever observed → conclude ONLY from a
  paginated-to-exhaustion `/orders/history` positive row (CANCELLED needs the
  row; everything else INCONCLUSIVE); grace ≥30 s flat; #41 shells excluded
  via the journal's child refs (never row-level exemption); #51-window scope
  on `cancel_siblings`-class writes, not confirm_missing.
- **Test plan (`plugins/dnse/tests/test_residue_detector.py`), red-first:**
  R1 vanished id + history CANCELLED row → residue event (today: silence);
  R2 vanished id + NO history row → INCONCLUSIVE forever (never CANCELLED);
  R3 page-0 blip / unreadable book → no residue verdict (feeds #54 ladder);
  R4 #41 shell with journalled child → child tracked, shell never a residue;
  R5 control: normal cancel via terminal transition → detector stays silent.
- **Hard questions:** where does the detector run (watch cycle vs reconcile);
  history pagination (#69's finding — drain against `total`); event surface
  (OrderEvent 'cancelled' vs a warning+halt?).

## Small cards (after B3, cheap)

- **#69** history single-page completeness (`len(rows) < total` → drain or
  UNKNOWN) — may be absorbed by B3's paginated-history work; if so, close as
  resolved-by with evidence.
- **#70** holiday-aware session phase (verification-dated VN holiday table in
  the shared phase function; `closed(holiday)`).

## Every firing ends with

- Working tree committed & pushed OR parked in backup/ with a card comment
  saying exactly where it stopped.
- The active card's thread reflects reality (a progress comment if mid-step).
- Suites green if anything was committed: `pytest plugins/dnse/tests/ -q` AND
  `pytest tests/ -q --ignore=tests/t00_pynecore/ast/test_045_lib_import_normalizer_invalid_alias.py`.
