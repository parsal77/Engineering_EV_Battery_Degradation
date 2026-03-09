# Problem definition

Dataset: NASA Battery Aging dataset

## Prediction target (one target only)
**Option B2: Remaining Useful Life (RUL)**

- **Target definition:** cycles remaining until discharge capacity falls below **80% of nominal capacity**.
- **Nominal capacity:** capacity measured during the first available discharge cycle for that battery (or the manufacturer nominal rating if provided; default to first discharge).
- **RUL at cycle _t_:** (index of first cycle where capacity < 0.8 * nominal) - t.
- **Units:** cycles.
- **Target availability:** only defined on discharge cycles with valid `Capacity` values.

## Scope and constraints
- Use time-respecting splits (no random mixing of cycles for the same battery).
- Use only information available up to cycle _t_ when predicting RUL.
- Treat missing/imputed `Capacity` carefully: if `Capacity` absent, do not fabricate RUL.
