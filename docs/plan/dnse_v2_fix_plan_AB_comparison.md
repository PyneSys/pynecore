# A/B comparison — two independent adversarial reviews of the same fix plan

**Question this answers** (operator, 2026-08-26): run the
`fix-plan-double-check-3-paths` skill a second time against the PRE-review plan
(`git show ed045b5:docs/plan/dnse_v2_fix_plan.md`, extracted to
`docs/plan/dnse_v2_fix_plan_V1_UNDER_REVIEW.md`), blind to round 1's verdicts,
and compare the two outputs to measure the review process's result quality.

- **Round 1**: card [#53](https://github.com/rubycell/pynecore/issues/53) —
  11 Opus workers, verdicts merged into plan v2 (commit `14927aa`).
- **Round 2**: card [#59](https://github.com/rubycell/pynecore/issues/59) —
  same skill, same 11 items, workers blinded to round 1 (plan source restricted
  to the V1 extract; repo-wide greps excluded `docs/plan/` after wave 1;
  upstream ROUND-2 verdicts injected wave-to-wave, never round-1's).
- Both rounds: falsePositive challenge → 3 serial-by-exclusion paths → per-path
  bad-fix vetting → select-among-vetted; R1 (cite or UNVERIFIED) + R2
  (red-first probes) discipline; every review leader-adjudicated on the card.

## Headline result

**The blind arm independently rediscovered the campaign's most serious defect.**
Round 2's item 7 found, with its own red-first probes, that DNSE's
`execute_cancel_with_outcome` returns `CANCEL_CONFIRMED` for a cancel that lost
to a fill — the exact mechanism round 1 carded as
[#55](https://github.com/rubycell/pynecore/issues/55) (HIGH: engine fires a
second MARKET → double-open). It also **replicated the measurement**: round 1
recorded 8.4 s of blocking verify on the shared event loop; round 2, blind,
measured 8.41 s (two-id shape) with a 4.26 s heartbeat gap vs 0.05 s control.
Two independent reviews, same defect, same number — that is the strongest
evidence the pipeline's findings are properties of the code, not of the
reviewer.

## Per-item verdict comparison

| # | Item | Round 1 (#53) | Round 2 (#59, blind) | Agreement |
|---|---|---|---|---|
| 0 | `_base.py` extraction | CONFIRMED on a refuted premise (no type checker sees the plugin; anti-shadow MRO guard) | REAL but mis-specified + mis-slotted; split 0a-now/0b-before-C; found a validator false-pass | **Convergent** core verdict; r2 adds sequencing split + validator defect |
| 1 | persist-first journal | CONFIRMED, worse than stated (lost reply not even parked; dead sentinel; persist-first at `_place`) | CONFIRMED + understated; 2 of the plan's rationales wrong (no `run_exit`; urllib3 not httpx); **timeout KILLS the run** (severity ↑) | **Convergent** incl. blind-matching defect list; fork on journal mechanism (below) |
| 2 | recovery + #51 inversion | PARTIAL FALSE POSITIVE (reference orphan passes never cancel live orders); two-class ladder; `metadata.ip` discovery | CONFIRMED w/ MAJOR correction — inversion premise false ×3; 2 new defects: page-0-only book reads, `--run-label` adoption hole | **Convergent** on the inversion being wrong; r2's two defects are unique catches |
| 3 | DisappearanceTracker | PROCEED, shape WRONG: union-visibility + immortal #41 shells no-op the ref set; in-app-cancel motivation false (22 events); grace 30 s flat | REFUTED as written: same union-blindness (probed, `calls=0`); `is_exempt` row-level can't rescue (probed); same false motivation; same wrong #51 hook; same 30 s grace | **Strongest blind convergence — 5 independent points match**; fork on surviving mechanism (below) |
| 4 | run-ownership | PARTIAL FALSE POSITIVE (40/20/40): identity-refusal already structural; owned-set protects bot from operator, not vice versa | RESCOPE ~80% FP: 4 raw-net engine consumers the clamp never reaches; restart hole runs the OPPOSITE direction (our fills unbooked); ownership can only BOUND a netting position | **Convergent direction, r2 deeper** (quantified + consumer enumeration + direction-inversion probe) |
| 5 | durable dedup cursor | Justification refuted; RECLASSED as hard prerequisite of Phase A; SDK `executions/{orderId}` `eventNo` | CONFIRMED w/ correction — co-requisite of item 1, not a today-fix; same `executions`/`eventNo` key | **Four-point convergence** (silent restarts, reclass, SDK endpoint, eventNo); minor fork on cursor source |
| 6 | netting accounting | SHRINK — 2/3 dissolved; survivor = `get_position` envelope-completeness (truncated page ≡ flat) | Partially dissolves (same 2/3); **escalation**: the L0 gate itself green-lights on a truncated read; fix shape CORRECTED — raise on PROVEN truncation only, never infer | **Convergent** shrink; r2 corrects r1's fix shape (cross-round correction) |
| 7 | cancel dispatch | ADOPT, RESCOPED; NEW HIGH DEFECT #55 (false CANCEL_CONFIRMED → double-open); 8.4 s blocking measured | RESCOPE — **blind rediscovery of #55** + measurement replication (8.41 s); 4 of the plan's prescriptions false positive; capitalcom artifact mis-cited (`submit_cancel` is bool, always True) | **Headline convergence** incl. independent measurement replication |
| 8 | read-error classification (#54) | PARTIAL FP — wrong mechanism: bare except never sees HTTP failures (18,936 silent polls measured); classify at status level | PARTLY CONFIRMED; gap narrower than planned (`/orders`-scoped); adds a stuck-read guard | **Convergent**; r2's stuck-read guard is a unique catch |
| 9 | broker_lab | PROCEED, resequenced: ~60% hollow; T16 red under runner = Phase A spec; DEFECT: 5 tests pin the anti-#51 retry (#58) | PROCEED-SPLIT (D0 now, pytest-hosted); T16 red BOTH halves; corpus rot on 3 trees; `fake_venue.get_instruments` shape defect. **ARM CONTAMINATED** (repo-wide grep leaked current-plan lines; disclosed, re-derived, excluded from independence scoring) | Convergent, but **excluded from independence claims** |
| 10 | WS feeds track (#50) | Both justifications refuted; fill-latency dead; TradingClient not production-grade (15 defects) | Premise survives; **both headline claims false positive — including the leader's own trading-WS probe** (lowercase channels vs documented UPPERCASE enum; venue ACKs bogus channels; no negative control) | **Cross-round correction of the leader** — #50 corrected, "control plane PROVEN" retracted |

## What the comparison says about review quality

**1. Blind replication rate is high on the money paths.** Items 0, 1, 3, 5, 6,
7, 8 all reached the same core verdict blind — with item 3 matching on five
independent points and item 7 matching on mechanism AND measurement. The
pipeline's verdicts on live-money code are reproducible, not reviewer noise.

**2. Each round still found things the other missed.** Round-2-unique: the
`--run-label` adoption hole; page-0-only book reads; the L0-gate escalation of
#57; the stuck-read guard; the run-dies-on-timeout severity; the
`is_exempt` row-level refutation probe; the capitalcom mis-citation; the
`fake_venue` shape defect; the validator false-pass. Round-1-unique:
`metadata.ip` ownership signal; the 22-event in-app-cancel corpus; #56 (VWAP
slice price); #58 (pinned anti-#51 retries); the TradingClient 15-defect audit.
**One pass of this pipeline is NOT exhaustive** — for critical plans, a second
blind round has real marginal yield.

**3. The process catches its own reviewers.** Round 2 refuted the leader's own
round-1 trading-WS probe (item 10: wrong channel casing, no negative control,
venue ACKs anything) and corrected round 1's item-6 fix shape (raise on proven
truncation only). Adversarial review composes — later rounds audit earlier
rounds, including the adjudicator.

**4. Design forks the two rounds left open** (need one decision each at
plan-merge time):
- **Item 1** — journal mechanism: r1 persist-first at the `_place` chokepoint
  vs r2's client-wrapper variant (see both comments).
- **Item 3** — surviving tracker role: r1 lifecycle refs (parent XOR child,
  B0 observe-only) vs r2 demote-to-residue-detector behind terminal-state
  transitions with history-only CANCELLED authority.
- **Item 5** — cursor source: r1 derive-on-restart vs r2 co-requisite-of-item-1
  persistence.
- **Item 4** — r2 splits the write-side asymmetry into its own item; r1 kept it
  as a residual alarm.

**5. Contamination is detectable and containable.** One arm of eleven (item 9)
leaked round-1 text through a repo-wide grep; the worker disclosed it,
re-derived independently, and the arm was excluded from independence scoring.
The blindness clause now excludes `docs/plan/` from searches (skill patched).

## Verdict-agreement scoreboard

| Class | Items | Count |
|---|---|---|
| Convergent core verdict, blind | 0, 1, 3, 5, 6, 7, 8 | 7 |
| Convergent direction, r2 materially deeper | 4 | 1 |
| Convergent + r2 corrects r1 (incl. the leader) | 6, 10 | 2 |
| Contaminated (excluded from scoring) | 9 | 1 |
| Divergent verdicts | — | 0 |

No item flipped verdict between rounds. The differences are depth, sequencing,
and unique catches — not contradictions.

## Follow-ups carded from round 2

- [#60](https://github.com/rubycell/pynecore/issues/60) — `--run-label` adoption hole (item 2 r2)
- [#61](https://github.com/rubycell/pynecore/issues/61) — page-0-only order-book reads (item 2 r2; sibling of #57's position-page blindness)
- [#62](https://github.com/rubycell/pynecore/issues/62) — L0 gate green-lights on truncated position read (item 6 r2; escalates #57, HIGH)
- [#63](https://github.com/rubycell/pynecore/issues/63) — `fake_venue.get_instruments` shape mismatch (item 9 r2)

(Each card links back to the #59 review comment that found it. Round-1's cards #54–#58 remain the fix queue of record; round 2
changed no severity downward, raised item 1's timeout to run-killing, and
confirmed #55's fix class — outcome fidelity, not blocked on Phase A.)
