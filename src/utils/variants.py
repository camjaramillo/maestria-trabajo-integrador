"""Utilities to define and materialize dataset variants for experiments.

Provides:
- VARIANT_DEFS: a declarative list of variant dictionaries (name, description, drop/keep rules, date_range, generation_keep)
- apply_variant(df, variant_def, ...): returns (df_variant, metadata)
- add_dia_festivo_from_holidays: adds DIA_SEMANA and FESTIVO columns (uses holidays.Colombia when available)

This file is intentionally conservative: it tolerates missing columns and returns metadata about applied transformations.
"""
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Known generation column candidates (common in the project)
GENERATION_CANDIDATES = ["TERMICA", "HIDRAULICA", "SOLAR", "COGENERADOR", "EOLICA"]

# Simple variant definitions. Adjust items as needed. Each variant is a dict with at least 'name'.

VARIANT_DEFS: List[Dict[str, Any]] = [
    # 1. df original (sin modificaciones)
    {"name": "v1_original",
     "description": "All variables from the dataset, no modifications",
     "add_calendar": False},
    {"name": "v1_original_lags",
     "description": "All variables + 24 lags of PRECIO",
     "add_calendar": False,
     "add_lags": 24},

    # 2. todas las variables + DIA_SEMANA y FESTIVO
    {"name": "v2_with_calendar",
     "description": "All variables + DIA_SEMANA and FESTIVO",
     "add_calendar": True},
    {"name": "v2_with_calendar_lags",
     "description": "All variables + DIA_SEMANA and FESTIVO + 24 lags of PRECIO",
     "add_calendar": True, "add_lags": 24},

    # 3. todas las variables excepto SOLAR
    {"name": "v3_no_solar",
     "description": "All variables except SOLAR generation",
     "drop_columns": ["SOLAR"]},
    {"name": "v3_no_solar_lags",
     "description": "All variables except SOLAR generation + 24 lags of PRECIO",
     "drop_columns": ["SOLAR"], "add_lags": 24},

    # 4. todas las variables excepto consumo de combustibles
    {"name": "v4_no_fuel_consumption",
     "description": "All variables except fuel consumption variables",
     "drop_columns": [
         "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
       "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP"
   ]},
    {"name": "v4_no_fuel_consumption_lags",
    "description": "All variables except fuel consumption variables + 24 lags of PRECIO",
    "drop_columns": [
        "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
        "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP"
    ], "add_lags": 24},

    # 5. todas las variables excepto consumo y costos de combustibles
    {"name": "v5_no_fuel_and_cost",
    "description": "All variables except fuel consumption and cost variables",
    "drop_columns": [
        "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
        "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP",
        "FUEL_COST_CARBON", "FUEL_COST_GAS", "FUEL_COST_GAS_NI", "FUEL_COST_COMBUSTOLEO"
    ]},
    {"name": "v5_no_fuel_and_cost_lags",
    "description": "All variables except fuel consumption and cost variables + 24 lags of PRECIO",
    "drop_columns": [
        "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
        "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP",
        "FUEL_COST_CARBON", "FUEL_COST_GAS", "FUEL_COST_GAS_NI", "FUEL_COST_COMBUSTOLEO"
    ], "add_lags": 24},

    # 6. todas las variables excepto indicadores económicos + consumo y costos de combustibles
    {"name": "v6_no_econ_fuel_cost",
    "description": "All variables except economic indicators, fuel consumption and cost variables",
    "drop_columns": [
        "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
        "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP",
        "FUEL_COST_CARBON", "FUEL_COST_GAS", "FUEL_COST_GAS_NI", "FUEL_COST_COMBUSTOLEO",
        "IPC_VAR_MOM_PCT", "IPP_VAR_PN_MOM_PCT", "IPP_VAR_OI_MOM_PCT"
    ]},
    {"name": "v6_no_econ_fuel_cost_lags",
    "description": "All variables except economic indicators, fuel consumption and cost variables + 24 lags of PRECIO",
    "drop_columns": [
        "FUEL_CONS_ACPM", "FUEL_CONS_CARBON", "FUEL_CONS_COMBUSTOLEO",
        "FUEL_CONS_CRUDO", "FUEL_CONS_GAS", "FUEL_CONS_GAS_NI", "FUEL_CONS_GLP",
        "FUEL_COST_CARBON", "FUEL_COST_GAS", "FUEL_COST_GAS_NI", "FUEL_COST_COMBUSTOLEO",
        "IPC_VAR_MOM_PCT", "IPP_VAR_PN_MOM_PCT", "IPP_VAR_OI_MOM_PCT"
    ], "add_lags": 24},

    # 7. solo variables de generación + ENSO
    {"name": "v7_only_gen_enso",
     "description": "Only generation variables + ENSO",
     "keep_only_generation": True, "keep_extra": ["NIVEL_ENSO"]},
    {"name": "v7_only_gen_enso_lags",
     "description": "Only generation variables + ENSO + 24 lags of PRECIO",
     "keep_only_generation": True, "keep_extra": ["NIVEL_ENSO"], "add_lags": 24},

    # 8. solo variables de generación (excepto SOLAR) + ENSO
    {"name": "v8_only_gen_no_solar_enso",
     "description": "Only generation variables (except SOLAR) + ENSO",
     "keep_only_generation": True, "drop_columns": ["SOLAR"], "keep_extra": ["NIVEL_ENSO"]},
    {"name": "v8_only_gen_no_solar_enso_lags",
     "description": "Only generation variables (except SOLAR) + ENSO + 24 lags of PRECIO",
     "keep_only_generation": True, "drop_columns": ["SOLAR"], "keep_extra": ["NIVEL_ENSO"], "add_lags": 24},

    # 9. todas las variables, rango de fechas 2024-01-01 a 2025-06-30
    {"name": "v9_date_range_all",
     "description": "All variables, restricted to 2024-01-01 to 2025-06-30",
     "date_range": ("2024-01-01", "2025-06-30")},
    {"name": "v9_date_range_all_lags",
     "description": "All variables, restricted to 2024-01-01 to 2025-06-30 + 24 lags of PRECIO",
     "date_range": ("2024-01-01", "2025-06-30"), "add_lags": 24},

    # 10. todas las variables excepto SOLAR, rango de fechas 2024-01-01 a 2025-06-30
    {"name": "v10_date_range_no_solar",
     "description": "All variables except SOLAR, restricted to 2024-01-01 to 2025-06-30",
     "drop_columns": ["SOLAR"], "date_range": ("2024-01-01", "2025-06-30")},
    {"name": "v10_date_range_no_solar_lags",
     "description": "All variables except SOLAR, restricted to 2024-01-01 to 2025-06-30 + 24 lags of PRECIO",
     "drop_columns": ["SOLAR"], "date_range": ("2024-01-01", "2025-06-30"), "add_lags": 24},
]




def _ensure_datetime_and_sort(df: pd.DataFrame, date_col: str = "FECHA_HORA") -> pd.DataFrame:
    if date_col not in df.columns:
        # Nothing to do; return copy
        return df.copy()
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        # If parsing fails, leave as-is
        pass
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def add_dia_festivo_from_holidays(df: pd.DataFrame, date_col: str = "FECHA_HORA", years: Optional[List[int]] = None) -> pd.DataFrame:
    """Add DIA_SEMANA (1=Monday..7=Sunday) and FESTIVO (1/0) using holidays.Colombia when available.

    Falls back to marking weekends as FESTIVO if holidays package not installed.
    """
    df = df.copy()
    df = _ensure_datetime_and_sort(df, date_col)
    if date_col not in df.columns:
        return df

    df["DIA_SEMANA"] = df[date_col].dt.weekday + 1

    try:
        import holidays

        years = years or sorted(list(set(df[date_col].dt.year.tolist())))
        try:
            col_holidays = holidays.Colombia(years=years)
            df["FESTIVO"] = df[date_col].dt.date.apply(lambda d: 1 if d in col_holidays else 0)
        except Exception:
            # fallback to weekend marking if holidays.Colombia fails
            df["FESTIVO"] = df[date_col].dt.weekday.apply(lambda w: 1 if w >= 5 else 0)
    except Exception:
        # holidays not installed: use weekend as FESTIVO
        df["FESTIVO"] = df[date_col].dt.weekday.apply(lambda w: 1 if w >= 5 else 0)

    return df


def detect_generation_columns_from_list(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> List[str]:
    candidates = candidates or GENERATION_CANDIDATES
    present = [c for c in candidates if c in df.columns]
    return present


def _parse_date_range(date_range: Optional[Tuple[str, str]]):
    if date_range is None:
        return None
    start, end = date_range
    try:
        start_ts = pd.to_datetime(start)
    except Exception:
        start_ts = None
    try:
        end_ts = pd.to_datetime(end)
    except Exception:
        end_ts = None
    return start_ts, end_ts


def apply_variant(
    df: pd.DataFrame,
    variant_def: Dict[str, Any],
    date_col: str = "FECHA_HORA",
    holiday_years: Optional[List[int]] = None,
    n_lags: int = 24,
    save_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the provided variant definition to the dataframe and return (df_variant, metadata).

    Supported variant_def keys:
    - name (required)
    - description (optional)
    - drop_columns: list of columns to drop if present
    - keep_columns: list of columns to keep (all others dropped)
    - keep_only_generation: if True, keep only generation columns (detected) + FECHA_HORA + PRECIO
    - drop_generation: if True, drop detected generation columns
    - add_lags: int number of lags to create for target (not implemented fully here; placeholder)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    vname = variant_def.get("name", "unnamed_variant")

    meta: Dict[str, Any] = {
        "name": vname,
        "description": variant_def.get("description", ""),
        "cols_dropped_missing": []
    }
    df0 = df.copy()
    meta["n_before"] = len(df0)

    # ensure datetime
    df0 = _ensure_datetime_and_sort(df0, date_col)

    # apply date range if provided
    date_range = variant_def.get("date_range")
    parsed = _parse_date_range(date_range)
    if parsed is not None:
        start_ts, end_ts = parsed
        if start_ts is not None:
            df0 = df0[df0[date_col] >= start_ts]
        if end_ts is not None:
            df0 = df0[df0[date_col] <= end_ts]
        meta["date_range_applied"] = (str(start_ts), str(end_ts))
    else:
        meta["date_range_applied"] = None

    # Add DIA_SEMANA and FESTIVO if requested
    if variant_def.get("add_calendar", True):
        df0 = add_dia_festivo_from_holidays(df0, date_col=date_col, years=holiday_years)

    # handle keep_columns (explicit)
    keep_cols = variant_def.get("keep_columns")
    if keep_cols:
        cols_exist = [c for c in keep_cols if c in df0.columns]
        missing = [c for c in keep_cols if c not in df0.columns]
        meta["cols_dropped_missing"].extend(missing)
        df0 = df0[cols_exist]
        meta["cols_used"] = cols_exist
        meta["n_after"] = len(df0)
        if save_csv is not None:
            try:
                Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
                df0.to_csv(save_csv, index=False)
                meta["saved_variant_csv"] = str(save_csv)
            except Exception:
                meta["saved_variant_csv"] = None
        return df0, meta

    # handle keep_only_generation (+ keep_extra)
    if variant_def.get("keep_only_generation"):
        gen_cols = detect_generation_columns_from_list(df0)
        cols_to_keep = [c for c in gen_cols if c in df0.columns]

        # siempre conservar fecha y target si existen
        for c in [date_col, "PRECIO"]:
            if c in df0.columns and c not in cols_to_keep:
                cols_to_keep.insert(0, c)

        # conservar columnas extra si se especifican
        keep_extra = variant_def.get("keep_extra", [])
        for c in keep_extra:
            if c in df0.columns and c not in cols_to_keep:
                cols_to_keep.append(c)

        df0 = df0[cols_to_keep]
        meta["cols_used"] = cols_to_keep
        meta["n_after"] = len(df0)
        if save_csv is not None:
            try:
                Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
                df0.to_csv(save_csv, index=False)
                meta["saved_variant_csv"] = str(save_csv)
            except Exception:
                meta["saved_variant_csv"] = None
        return df0, meta

    # handle drop_generation
    if variant_def.get("drop_generation"):
        gen_cols = detect_generation_columns_from_list(df0)
        for c in gen_cols:
            if c in df0.columns:
                df0 = df0.drop(columns=[c])

    # drop explicit columns
    drop_cols = variant_def.get("drop_columns", []) or []
    actually_dropped = [c for c in drop_cols if c in df0.columns]
    missing = [c for c in drop_cols if c not in df0.columns]
    meta["cols_dropped_missing"].extend(missing)
    if actually_dropped:
        df0 = df0.drop(columns=actually_dropped)

    # Optionally add lags
    add_lags = variant_def.get("add_lags")
    if add_lags and ("PRECIO" in df0.columns):
        lag_n = int(add_lags)
        for lag in range(1, lag_n + 1):
            colname = f"PRECIO_lag_{lag}"
            df0[colname] = df0["PRECIO"].shift(lag)
        df0 = df0.dropna().reset_index(drop=True)

    meta["cols_used"] = [c for c in df0.columns]
    meta["n_after"] = len(df0)

    if save_csv is not None:
        try:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df0.to_csv(save_csv, index=False)
            meta["saved_variant_csv"] = str(save_csv)
        except Exception:
            meta["saved_variant_csv"] = None

    return df0, meta

