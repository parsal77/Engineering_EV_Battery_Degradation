Dataset: NASA Battery Aging dataset

# Design note

## Objective
Build a leakage-safe pipeline for EV battery degradation modeling using cycle-level data.

## Key decisions
- Target: Remaining Useful Life (cycles to 80% nominal capacity) if computable, otherwise capacity prediction.
- Splits: time-respecting within-battery and battery-level holdout.
- Baselines: simple slope heuristic + linear regression.

## Plan
1. Ingest and validate schema (battery_id, cycle, timestamp, current, voltage, temperature, capacity).
2. Define target label without peeking into future cycles.
3. Build leakage-safe splits and baselines.
4. Engineer stress + trend features (thermal/electrical load, rolling capacity slope) that reflect degradation mechanisms.
5. Train non-linear model and evaluate honestly with error vs cycle count.
