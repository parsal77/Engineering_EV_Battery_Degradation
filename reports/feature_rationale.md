# Feature Rationale

## Signal Families
- Electrical: voltage extrema, range, energy discharged.
- Thermal: max temperature and in-cycle temperature rise.
- Temporal: discharge duration and cycle index effects.
- Degradation memory: lagged capacity and rolling fade-rate features.

## Engineering Intent
- Capture immediate cycle behavior (`max_voltage`, `internal_resistance_proxy`).
- Capture cumulative aging trend (`capacity_fade_rate`, lag features).
- Capture efficiency-related stress (`cycle_efficiency`).

## Practical Relevance
These features map to concepts used by battery management systems (BMS): state tracking, stress monitoring, and degradation trend estimation.
