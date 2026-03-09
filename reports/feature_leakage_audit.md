# Feature Leakage Audit

This audit documents feature timing and leakage risk for SOH/RUL prediction.

## Evaluation Context
- Holdout battery: `B0018`
- Train batteries: `B0005`, `B0006`, `B0007`
- Temporal order preserved within each battery.

## Leakage Guard Rules
- Target column is never allowed as an input feature.
- For SOH prediction, `capacity_Ah` is explicitly forbidden as an input.
- Scalers are fit on train batteries only.

## Feature-Level Audit

| Feature | Available at cycle end? | Uses future cycles? | Leakage risk | Notes |
|---|---|---|---|---|
| `cycle_number` | Yes | No | Low | Sequence progress indicator. |
| `avg_voltage` | Yes | No | Low | Aggregated current-cycle measurement. |
| `avg_current` | Yes | No | Low | Aggregated current-cycle measurement. |
| `avg_temperature` | Yes | No | Low | Aggregated current-cycle measurement. |
| `charge_time` | Yes | No | Low | Previous charge profile summary. |
| `discharge_time` | Yes | No | Low | Current discharge duration. |
| `energy_charged_Wh` | Yes | No | Medium | Cycle summary from charge profile. |
| `max_voltage` | Yes | No | Low | Derived from current cycle. |
| `min_voltage` | Yes | No | Low | Derived from current cycle. |
| `voltage_range` | Yes | No | Low | Derived from current cycle. |
| `max_temperature` | Yes | No | Low | Derived from current cycle. |
| `temperature_rise` | Yes | No | Low | Derived from current cycle. |
| `discharge_duration` | Yes | No | Low | Equivalent to cycle duration summary. |
| `energy_discharged` | Yes | No | Medium | Directly linked to cycle behavior. |
| `internal_resistance_proxy` | Yes | No | Low | Early-cycle slope proxy. |
| `capacity_fade_rate` | Yes | No | Medium | Uses trailing 5-cycle history only. |
| `cycle_efficiency` | Yes | No | Medium | Ratio of in-cycle energy terms. |
| `capacity_lag_1` | Yes | No | Medium | Uses prior cycle only. |
| `capacity_lag_3` | Yes | No | Medium | Uses historical lag only. |
| `capacity_lag_5` | Yes | No | Medium | Uses historical lag only. |
| `capacity_Ah` (SOH task) | Yes | No | **High (forbidden)** | Trivial proxy for SOH (`SOH = capacity_Ah / 2.0`). Excluded by guard. |

## Conclusion
Current code enforces leakage-safe feature selection for SOH via explicit validation. Any attempt to include `capacity_Ah` in SOH features now raises an error.
