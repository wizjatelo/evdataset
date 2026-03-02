# EV-HybridDataset-2024: Hybrid Electric Vehicle Dataset for AI Model Training
## Charging Behavior | Battery Degradation | Energy Consumption

**Version:** 1.0.0  
**License:** CC BY 4.0  
**Competition:** IEEE PES & DataPort – Good Datasets for AI Model Training in Power and Energy Domain  
**Contact:** [wisdomarkanyange@gmail.com]

---

## Overview

This dataset package provides a comprehensive, AI-ready hybrid dataset covering three interconnected domains of electric vehicle (EV) operation. Real-world parameters and distributions are drawn from publicly available sources and calibrated against peer-reviewed literature; identified data gaps are filled through physics-informed stochastic synthesis, fully documented and flagged with a `data_source` field in every table.

---

## Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `ev_charging_behavior.csv` | 50,000 | 18 | ~7 MB | Charging session logs |
| `ev_battery_degradation.csv` | 37,506 | 15 | ~5 MB | Cell-level SOH degradation records |
| `ev_energy_consumption.csv` | 80,000 | 18 | ~10 MB | Per-trip energy consumption |
| `README.md` | — | — | — | This file |

---

## Data Sources

### Real-World Calibration Bases
| Source | Domain | Records |
|--------|--------|---------|
| EPFL DESL Level-3 EV Charging Dataset (Switzerland, 2022–2023) | Charging | ~14 months of DC session data |
| UrbanEV / ST-EVCDP – Shenzhen, China (2022–2023) | Charging demand | 1,682 stations, 24,798 piles |
| Boulder CO Open Data Portal (2023) | Charging / Energy | 148,136 anonymized sessions |
| NASA Li-ion Battery Aging Datasets (PCoE) | Degradation | 18650 cells, CC discharge profiles |
| CALCE Battery Group – UMD (CS2 dataset) | Degradation | 15 LCO prismatic cells |
| NMC/C-SiO Aging Dataset – Scientific Data (2024) | Degradation | 228 cells, >3B data points |
| emobpy tool – Scientific Data (2021) | Energy consumption | 200 BEV profiles, Germany |
| MDPI Energies – Thailand BEV study (2023) | Energy consumption | Real-world OBD data |

### Synthesized Gap-Filling (flagged `data_source = synthetic`)
- Longitudinal per-vehicle SOH tracking (linked vehicle ID across charging + degradation)
- EIS impedance data (1 kHz) at intermediate SOH levels
- Full trip metadata with weather + road type + HVAC combinations
- User behavioral archetypes (opportunistic, scheduled, range-anxious, commuter)
- Cold-weather fast-charging sessions below -10 °C (safety-critical edge cases)

---

## Dataset 1: ev_charging_behavior.csv

### Variable Dictionary
| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| session_id | string | — | — | Unique session identifier (CH0000001…) |
| vehicle_id | string | — | — | Anonymized vehicle ID (V00001…V05000) |
| user_id | string | — | — | Anonymized user ID (U00001…U08000) |
| session_start | datetime | YYYY-MM-DD HH:MM:SS | 2022–2024 | Session start timestamp |
| session_end | datetime | YYYY-MM-DD HH:MM:SS | 2022–2024 | Session end timestamp |
| charge_type | categorical | — | AC_L1, AC_L2, DC_Fast | Charger level |
| connector_type | categorical | — | J1772, Type2, CCS, CHAdeMO, Tesla_DC | Connector standard |
| charger_power_kW | float | kW | 1.4–350 | Rated charger output power |
| energy_delivered_kWh | float | kWh | 0.1–150 | Total energy delivered to vehicle |
| session_duration_min | float | minutes | 2–720 | Session duration |
| soc_start_pct | float | % | 5–90 | Battery SOC at session start |
| soc_end_pct | float | % | 7–100 | Battery SOC at session end |
| battery_capacity_kWh | int | kWh | 40, 50, 58, 64, 75, 82, 100 | Vehicle nominal battery capacity |
| location_type | categorical | — | home, workplace, public_urban, highway_corridor | Charging location type |
| ambient_temp_C | float | °C | −20 to 45 | Ambient air temperature |
| charging_cost_USD | float | USD | 0–90 | Estimated session cost |
| user_archetype | categorical | — | opportunistic, scheduled, range_anxious, commuter | Behavioral archetype |
| data_source | categorical | — | real_calibrated, synthetic | Data origin flag |

---

## Dataset 2: ev_battery_degradation.csv

### Variable Dictionary
| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| vehicle_id | string | — | — | Anonymized vehicle ID |
| cell_chemistry | categorical | — | NMC, LFP, NCA | Li-ion cell chemistry |
| cycle_number | int | — | 0–1200 | Full equivalent cycle count at checkup |
| calendar_age_days | float | days | 0–3000 | Days in service at checkup |
| soh_pct | float | % | 86–100 | State of Health (capacity-based) |
| capacity_Ah | float | Ah | — | Measured usable capacity at checkup |
| nominal_capacity_kWh | float | kWh | 40–100 | Vehicle nominal pack capacity |
| internal_resistance_mOhm | float | mΩ | 5–120 | DC internal resistance |
| eis_impedance_1kHz_Ohm | float | Ω | — | EIS magnitude at 1 kHz (synthesized) |
| voltage_OCV_V | float | V | 3.2–4.2 | Open-circuit voltage at full charge |
| dod_pct | float | % | 0–100 | Depth of discharge during typical cycle |
| c_rate_charge | float | C | 0.3–3.0 | Charging C-rate |
| c_rate_discharge | float | C | 0.5–2.5 | Discharge C-rate |
| temperature_C | float | °C | −10 to 55 | Cell temperature during cycling |
| data_source | categorical | — | real_calibrated, synthetic | Data origin flag |

---

## Dataset 3: ev_energy_consumption.csv

### Variable Dictionary
| Column | Type | Unit | Range | Description |
|--------|------|------|-------|-------------|
| trip_id | string | — | — | Unique trip identifier |
| vehicle_id | string | — | — | Anonymized vehicle ID |
| vehicle_model | categorical | — | 6 models | Vehicle make/model/capacity tag |
| battery_capacity_kWh | int | kWh | 40–100 | Nominal battery capacity |
| road_type | categorical | — | urban, highway, mixed, rural | Predominant road type |
| trip_distance_km | float | km | 0.5–600 | Total trip distance |
| avg_speed_kmh | float | km/h | 5–180 | Average trip speed |
| ambient_temp_C | float | °C | −20 to 45 | Ambient temperature |
| hvac_active | int | bool | 0, 1 | HVAC system on (1) or off (0) |
| elevation_gain_m | float | m | 0–600 | Net elevation gain for trip |
| payload_kg | float | kg | 0–400 | Additional payload (passengers + cargo) |
| gross_energy_consumed_kWh | float | kWh | 0.1–85 | Total energy drawn from battery |
| regenerative_braking_kWh | float | kWh | 0–20 | Energy recovered via regeneration |
| net_energy_consumed_kWh | float | kWh | 0.1–80 | Net energy consumed (gross − regen) |
| consumption_rate_kWh100km | float | kWh/100 km | 8–45 | Normalized energy consumption rate |
| soc_before_trip_pct | float | % | 15–100 | SOC before trip |
| soc_after_trip_pct | float | % | 1–99 | SOC after trip |
| data_source | categorical | — | real_calibrated, synthetic | Data origin flag |

---

## Metadata

```yaml
dataset_name: EV-HybridDataset-2024
version: 1.0.0
created: 2024-12-01
license: CC BY 4.0
language: en
keywords: [electric vehicle, charging behavior, battery degradation, energy consumption,
           state of health, AI training, power systems, hybrid dataset]
temporal_coverage: 2022-01-01 / 2024-12-31
spatial_coverage: Global (North America, Europe, Asia – calibrated from multi-regional sources)
primary_use_cases:
  - EV charging load forecasting
  - Battery SOH estimation and remaining useful life prediction
  - Energy consumption modeling and range prediction
  - User behavior clustering
  - V2G planning and grid integration
AI_readiness:
  train_test_split_column: data_source
  recommended_split: 80/20 (train/test)
  target_variables:
    charging: energy_delivered_kWh, session_duration_min
    degradation: soh_pct, remaining_useful_life (derived)
    consumption: consumption_rate_kWh100km, net_energy_consumed_kWh
  missing_values: none
  normalization_required: yes (numerical columns)
  encoding_required: yes (categorical columns)
```

---

## Identified Data Gaps and Synthesis Rationale

| Gap | Why it Exists | How We Filled It |
|-----|--------------|------------------|
| Longitudinal per-vehicle SOH tracking | Public charging datasets don't link sessions to individual battery health | Vehicle-ID linkage + physics-informed SOH model |
| EIS data in field conditions | Lab datasets have EIS; field datasets don't | Calibrated from NMC/C-SiO dataset, extrapolated via equivalent circuit model |
| Cold-weather DC fast charging | EPFL dataset is Switzerland (mild) | Gaussian noise extension + temperature-power derating model |
| Behavioral archetypes | Raw transaction data has no user intent labels | K-means-inspired probabilistic assignment based on SOC start + location + time patterns |
| Diverse road types + weather | Boulder data is urban; emobpy is German highways | Cross-product sampling with physics-adjusted consumption rates |

---

## Validation Summary

| Dataset | Missing Values | Range Violations | Logic Violations |
|---------|---------------|-----------------|-----------------|
| Charging Behavior | 0 | 0 | 0 (soc_end > soc_start enforced) |
| Battery Degradation | 0 | 0 (SOH capped 100%) | 0 |
| Energy Consumption | 0 | 0 | 0 (soc_after < soc_before enforced) |

---

## Citation

If you use this dataset, please cite:
> [Author(s)]. (2024). *EV-HybridDataset-2024: A Hybrid Electric Vehicle Dataset for AI Model Training in Charging Behavior, Battery Degradation, and Energy Consumption*. IEEE DataPort. DOI: [pending]

And the underlying real-world sources listed in the Sources section above.

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.
