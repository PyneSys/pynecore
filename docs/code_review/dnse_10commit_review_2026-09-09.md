# Adversarial review — dnse-broker-v2 HEAD~10..HEAD (42f1f05), 2026-09-09

Independent Opus reviewer over the 10-commit range (#81, #85, #86, #87, T19-21
probes, grades, #91, #90/#92), instructed to hunt CROSS-COMMIT interactions —
each commit had its own panel; the composition had never been reviewed.
Probes were red-first with base-tree (ea81f24) worktree controls. Full probe
outputs live in the session transcript; findings are carded #93-#99.

| # | Sev | Finding | Card |
|---|-----|---------|------|
| 1 | CRITICAL | Bracket (TP+SL) exit modify = silent fabricated success: OCO child tracked NORMAL → `_amend_normal`; `_intent_price` picks TP; trailing-SL-only change → zero payloads → synthesized success, no warning, SL never reaches the wire (probe: 0 PUTs vs 1 at base). #85's exit park never fires for brackets; `_modify_warned_keys` never cleared for exits. | #93 |
| 2 | HIGH | `tools/flatten.py` attribution unscoped by account AND day; live store holds 3 account identities; NORMAL id space REUSED across days (09-08 issued lower ids than 09-07) → a stale row can claim the operator's live order. | #96 |
| 3 | HIGH | EXPIRED→'cancelled' (#87) with no `cancel_reason` → unexpected-cancel policy (default "stop") → probe-measured QUARANTINE on an ordinary 14:45 expiry (pre-#87 control: no quarantine). | #94 |
| 4 | HIGH | #87 prune fires on ALREADY_FILLED (contradicting `_scan_row`'s own #55 exemption) → severs `_adopt_child`'s `parent_id in ids` join (child never enters key scope) and degrades the next ask to UNKNOWN → cancel_tentative stall over an open position. | #95 |
| 5 | MED | `_amend_normal` blocks the event loop (3-6 sync venue round-trips, no `to_thread`); probe: 0 loop ticks during a modify. Repo already pins this invariant for the cancel path. | #97 |
| 6 | MED | (folded into #93) exit park keyed on recorded book, not leg semantics. | #93 |
| 7 | MED | (folded into #93) warned-keys process-lifetime for exits vs per-episode docstring. | #93 |
| 8 | MED-LOW | feed_timeout_bars=16 clears the ATC gap by 0-4 s at 1m — L4-T03 measured the gap at 16 min (last delivered bar = the 14:28 slot ~14:29), not 15; threshold control pins the wrong floor (>=16 should be >=17). | #98 |
| 9 | LOW | `entry_stop_limit_native` unvalidated by the plugin contract (latent; DNSE honours it, T21-proven). | #99 |

Also: the documented full-suite command hits the pre-existing upstream math.log
red under pytest.ini's `-x` (run stops early) — CLAUDE.md now carries the
deselect (fixed in this commit).

Seams walked CLEAN (with the walk described): #85 replace vs #86 detail-read
(mutually exclusive by book; prune-before-append keeps ids[0] the successor);
`_order_ids` consumers vs the #87 prune (only `_adopt_child` breaks — finding
4); `entry_stop_limit_native` watch consumers (all self-guard on missing
watch; no orphan watch-row is created); flatten's direct
`_cancel_one_disposition` on an empty-map standalone broker (no assumption
broken; `STILL_OPEN` in `_RESOLVED` is correct per models.py); the watchdog
outside ATC (clock genuinely pauses off-session); #83's suppressor ordering;
test hygiene (no weakened assertions found — gaps are missing coverage, not
soft pins).
