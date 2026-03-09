# Data dictionary

Dataset: NASA Battery Aging dataset

## Top-level cycle fields

| Column | Meaning | Units | Typical range | Missingness |
|---|---|---|---|---|
| cycle | list/array of cycle records | n/a | depends on dataset | never |
| type | cycle type label: `charge`, `discharge`, `impedance` | categorical | charge/discharge/impedance | never |
| ambient_temperature | ambient temperature during cycle | °C | ~0–60 | never |
| time | start time of cycle (MATLAB datenum vector) | datetime vector | monotonic increasing | never |
| data | nested structure containing measured signals for that cycle | n/a | n/a | never |

## Charge cycle (data fields)

| Column | Meaning | Units | Typical range | Missingness |
|---|---|---|---|---|
| Voltage_measured | measured terminal voltage | V | 0–4.3 | only present for charge cycles |
| Current_measured | measured current | A | 0–5 (positive for charge) | only present for charge cycles |
| Temperature_measured | measured cell temperature | °C | ~0–60 | only present for charge cycles |
| Current_charge | commanded/charge current setpoint | A | 0–5 | only present for charge cycles |
| Voltage_charge | commanded voltage setpoint | V | 0–4.3 | only present for charge cycles |
| Time | elapsed time within cycle | s | 0–20000 | only present for charge cycles |

## Discharge cycle (data fields)

| Column | Meaning | Units | Typical range | Missingness |
|---|---|---|---|---|
| Voltage_measured | measured terminal voltage | V | 2.0–4.2 | only present for discharge cycles |
| Current_measured | measured current | A | 0–5 (positive for discharge) | only present for discharge cycles |
| Temperature_measured | measured cell temperature | °C | ~0–60 | only present for discharge cycles |
| Current_charge | commanded current setpoint (during charge before discharge) | A | 0–5 | only present for discharge cycles |
| Voltage_charge | commanded voltage setpoint (during charge before discharge) | V | 0–4.3 | only present for discharge cycles |
| Time | elapsed time within cycle | s | 0–20000 | only present for discharge cycles |
| Capacity | discharge capacity recorded at end of discharge (to 2.7V cutoff) | Ah | 0–2 | only present for discharge cycles |

## Impedance cycle (data fields)

| Column | Meaning | Units | Typical range | Missingness |
|---|---|---|---|---|
| Sense_current | sense current | A | small (<1) | only present for impedance cycles |
| Battery_current | battery current | A | small (<1) | only present for impedance cycles |
| Current_ratio | ratio of currents | unitless | ~0–1 | only present for impedance cycles |
| Battery_impedance | battery impedance magnitude | Ohm | milliohm range | only present for impedance cycles |
| Rectified_impedance | rectified impedance | Ohm | milliohm range | only present for impedance cycles |
| Re | electrochemical resistance estimate | Ohm | milliohm range | only present for impedance cycles |
| Rct | charge transfer resistance estimate | Ohm | milliohm range | only present for impedance cycles |

Notes:
- The dataset is structured and field presence depends on cycle `type`; treat absent fields as structurally missing (not measurement dropouts).
- Typical ranges are approximate and should be refined after ingest.
