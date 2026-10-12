# Reusing invariant features in a parameter sweep

When only downstream parameters change, a closed-bar feature can be computed once
and supplied through each candle's `extra_fields`. This is an application-level
pattern using existing `ScriptRunner` APIs. It does not modify the engine or
guarantee a speedup. Feature storage increases memory and has a preparation cost.

The accompanying example uses96 generated synthetic candles, a7bar SMA and three
arithmetic offsets. It contains no real price history or trading strategy.
Run `python test_feature_reuse.py` with PyneCore installed. The named
`__test_synthetic_feature_reuse__` function follows upstream's pytest function
naming convention; its final category/path requires maintainer agreement.

`feature_direct.py` calculates SMA plus an offset on each run. `feature_source.py`
calculates SMA once. `feature_cached.py` reads that aligned value and applies the
offset. The driver constructs new candles rather than mutating original candles.
It copies scalar plot values immediately when using `copy_results=False`.
Keeping the yielded mutable dictionaries instead would corrupt retained output.

Validate equivalence before measuring: same number/order of bars, exact values,
warmup missing-value masks and causal prefix output. The test also demonstrates
that reusing a7bar SMA for an11bar configuration produces different results.
Changing feature parameters must recompute features or use the full calculation.

This example is deliberately limited to closed standard1minute bars with a fixed
synthetic symbol. A production cache identity must cover data content/order,
feature source and runtime version, all feature-affecting inputs, symbol metadata,
timeframe/session/calendar/timezone, adjustments/rollovers and warmup/NA policy.
Never reuse a feature that depends on varied parameters, broker positions,
order fills, evolving realtime candles, MTF context or future bars without a
separate correctness argument. Reject misalignment; don't silently truncate.

For an honest benchmark, include feature preparation, use identical cases,
interleave baseline/candidate repeats, record peak memory, and measure without a
profiler. Keep process startup/data loading/serialization inclusion consistent.
No performance percentage is asserted by this small correctness example.

Proposed upstream location: a linked example under `docs/programmatic/`;
three scripts plus the synthetic check can live in `scripts/examples/feature_reuse/`
or a maintainer-selected equivalent. New material is proposed under Apache-2.0.
