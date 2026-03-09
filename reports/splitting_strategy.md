# Splitting Strategy

## Objective
Evaluate battery health prediction performance without leakage and with realistic generalization constraints.

## Leakage Prevention Protocol
1. No random split across cycles from the same battery.
2. Battery-level holdout split for generalization testing (`B0018` held out).
3. Temporal ordering preserved inside each battery sequence.
4. Lag features computed using prior-cycle information only.
5. Scalers fit on training batteries only and applied to holdout battery.

## Why This Matters
Cycle-level random shuffling can leak degradation trajectory information into the test set and inflate metrics. Battery-level holdout better reflects deployment to unseen packs.
