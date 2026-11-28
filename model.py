"""Motor Monte Carlo multiperiodo y estimación descriptiva de distribuciones."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


CONTINUOUS = ("Normal", "Lognormal", "Gamma", "Exponencial", "Uniforme", "Triangular", "Empírica")
FREQUENCY = ("Poisson", "Binomial", "Fija", "Empírica")


def numeric_history(frame: pd.DataFrame, col: str | None, *, counts: bool = False) -> np.ndarray | None:
    if col is None or col not in frame.columns:
        return None
    data = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if not len(data):
        return None
    if counts and (np.any(data < 0) or np.any(data != np.floor(data))):
        raise ValueError("La columna de frecuencia requiere enteros no negativos.")
    return data


def fit_severity(data: np.ndarray) -> pd.DataFrame:
    """Comparación AIC (densidad) de familias continuas, sin afirmar prueba de ajuste."""
    if len(data) < 5 or np.std(data) == 0:
        raise ValueError("Se requieren al menos 5 datos no constantes para comparar distribuciones.")
    specs = {
        "Normal": (stats.norm, {}),
        "Uniforme": (stats.uniform, {}),
    }
    if np.all(data > 0):
        specs.update({
            "Lognormal": (stats.lognorm, {"floc": 0}),
            "Gamma": (stats.gamma, {"floc": 0}),
            "Exponencial": (stats.expon, {"floc": 0}),
        })
    rows = []
    for name, (distribution, options) in specs.items():
        try:
            pars = distribution.fit(data, **options)
            log_pdf = distribution.logpdf(data, *pars)
            if not np.all(np.isfinite(log_pdf)):
                continue
            k = len(pars) - len(options)
            aic = 2 * k - 2 * np.sum(log_pdf)
            rows.append({"Distribución": name, "AIC": float(aic), "Parámetros SciPy": str(tuple(round(float(v), 6) for v in pars))})
        except (ValueError, FloatingPointError):
            pass
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def estimate_parameters(data: np.ndarray, distribution: str) -> dict:
    if distribution == "Normal":
        return {"media": float(np.mean(data)), "desv": max(float(np.std(data, ddof=1)), 0.000001)}
    if distribution == "Uniforme":
        lo, hi = float(np.min(data)), float(np.max(data))
        return {"min": lo, "max": hi if hi > lo else lo + 1e-6}
    if distribution == "Triangular":
        lo, hi = float(np.min(data)), float(np.max(data))
        return {"min": lo, "moda": float(np.median(data)), "max": hi if hi > lo else lo + 1e-6}
    if distribution in ("Lognormal", "Gamma", "Exponencial"):
        if np.any(data <= 0):
            raise ValueError(f"La distribución {distribution} requiere todos los valores históricos positivos.")
        if distribution == "Lognormal":
            z = np.log(data)
            return {"mu_log": float(z.mean()), "sigma_log": max(float(z.std(ddof=1)), 1e-6)}
        if distribution == "Gamma":
            a, _, scale = stats.gamma.fit(data, floc=0)
            return {"forma": float(a), "escala": float(scale)}
        return {"escala": float(np.mean(data))}
    return {}


def severity_draw(rng: np.random.Generator, distribution: str, params: dict, historical: np.ndarray | None, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    p = params
    if distribution == "Empírica":
        if historical is None:
            raise ValueError("Seleccione una columna histórica de severidad.")
        return rng.choice(historical, size=n, replace=True)
    if distribution == "Normal":
        return rng.normal(p["media"], p["desv"], n)
    if distribution == "Lognormal":
        return rng.lognormal(p["mu_log"], p["sigma_log"], n)
    if distribution == "Gamma":
        return rng.gamma(p["forma"], p["escala"], n)
    if distribution == "Exponencial":
        return rng.exponential(p["escala"], n)
    if distribution == "Uniforme":
        if p["max"] <= p["min"]:
            raise ValueError("Uniforme: máximo debe superar al mínimo.")
        return rng.uniform(p["min"], p["max"], n)
    if distribution == "Triangular":
        if not p["min"] < p["max"] or not p["min"] <= p["moda"] <= p["max"]:
            raise ValueError("Triangular: mínimo < máximo y mínimo ≤ moda ≤ máximo.")
        return rng.triangular(p["min"], p["moda"], p["max"], n)
    raise ValueError("Distribución de severidad no válida.")


def frequency_draw(rng: np.random.Generator, distribution: str, params: dict, historical: np.ndarray | None, n: int) -> np.ndarray:
    if distribution == "Empírica":
        if historical is None:
            raise ValueError("Seleccione una columna histórica de frecuencia.")
        return rng.choice(historical.astype(np.int64), size=n)
    if distribution == "Poisson":
        return rng.poisson(params["lambda"], n)
    if distribution == "Binomial":
        return rng.binomial(params["ensayos"], params["p"], n)
    if distribution == "Fija":
        return np.full(n, params["eventos"], dtype=np.int64)
    raise ValueError("Distribución de frecuencia no válida.")


def seasonality_index(frame: pd.DataFrame, date_col: str, value_col: str) -> np.ndarray:
    """Índices mensuales relativos (media 1), usando suma de valores por mes/año."""
    sample = frame[[date_col, value_col]].copy()
    sample[date_col] = pd.to_datetime(sample[date_col], errors="coerce")
    sample[value_col] = pd.to_numeric(sample[value_col], errors="coerce")
    sample = sample.dropna()
    if len(sample) < 12 or sample[date_col].dt.to_period("M").nunique() < 12:
        raise ValueError("La estimación estacional necesita 12 meses diferentes como mínimo.")
    monthly = sample.groupby(sample[date_col].dt.to_period("M"))[value_col].sum()
    baseline = float(monthly.mean())
    if baseline <= 0:
        raise ValueError("La estacionalidad multiplicativa requiere media histórica mensual positiva.")
    by_month = monthly.groupby(monthly.index.month).mean()
    indices = np.ones(12, dtype=float)
    for month, value in by_month.items():
        indices[int(month) - 1] = max(float(value) / baseline, 0.)
    return indices / indices.mean()


def historical_trend(frame: pd.DataFrame, date_col: str, value_col: str) -> float:
    """Cambio medio porcentual por mes, estimado sobre serie mensual agregada positiva."""
    sample = frame[[date_col, value_col]].copy()
    sample[date_col] = pd.to_datetime(sample[date_col], errors="coerce")
    sample[value_col] = pd.to_numeric(sample[value_col], errors="coerce")
    sample = sample.dropna()
    monthly = sample.groupby(sample[date_col].dt.to_period("M"))[value_col].sum().sort_index()
    if len(monthly) < 6 or np.any(monthly <= 0):
        raise ValueError("Se requieren 6 meses con valores agregados positivos para estimar tendencia.")
    x = (monthly.index.year - monthly.index[0].year) * 12 + monthly.index.month - monthly.index[0].month
    slope = np.polyfit(np.asarray(x, float), np.log(monthly.to_numpy(float)), 1)[0]
    return float(np.expm1(slope))


def project(*, iterations: int, horizon: int, seed: int, aggregated: bool,
            sev_dist: str, sev_params: dict, sev_hist: np.ndarray | None,
            freq_dist: str, freq_params: dict, freq_hist: np.ndarray | None,
            monthly_growth: float = 0., seasonal: np.ndarray | None = None,
            start_month: int = 1, max_draws: int = 8_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve matrices [escenario, periodo] de valor y número de eventos.

    La tendencia y estacionalidad multiplican la severidad (no el conteo).
    Los periodos son independientes condicionalmente a parámetros y factores.
    """
    if not (1 <= iterations <= 100_000 and 1 <= horizon <= 120):
        raise ValueError("Fuera de rango: 1–100,000 escenarios; 1–120 periodos.")
    rng = np.random.default_rng(seed)
    count = iterations * horizon
    if aggregated:
        frequencies = frequency_draw(rng, freq_dist, freq_params, freq_hist, count)
        if np.any(frequencies < 0):
            raise ValueError("Las frecuencias deben ser no negativas.")
    else:
        frequencies = np.ones(count, dtype=np.int64)
    total_events = int(np.sum(frequencies, dtype=np.int64))
    if total_events > max_draws:
        raise ValueError(f"Se requieren {total_events:,} eventos, sobre el límite de {max_draws:,}. Reduzca escenarios, horizonte o frecuencia.")
    draws = severity_draw(rng, sev_dist, sev_params, sev_hist, total_events)
    if not np.all(np.isfinite(draws)):
        raise ValueError("Los parámetros generan resultados no finitos.")
    values = np.zeros(count, dtype=float)
    if total_events:
        np.add.at(values, np.repeat(np.arange(count), frequencies), draws)
    values = values.reshape(iterations, horizon)
    frequencies = frequencies.reshape(iterations, horizon)
    factors = np.power(1 + monthly_growth, np.arange(1, horizon + 1))
    if seasonal is not None:
        if len(seasonal) != 12:
            raise ValueError("La estacionalidad debe tener doce factores mensuales.")
        months = (start_month - 1 + np.arange(horizon)) % 12
        factors *= seasonal[months]
    values *= factors[None, :]
    if not np.all(np.isfinite(values)):
        raise ValueError("La proyección generó valores no finitos.")
    return values, frequencies


def result_statistics(values: np.ndarray, confidence: float) -> pd.DataFrame:
    rows = []
    alpha = 1 - confidence
    for period in range(values.shape[1]):
        sample = values[:, period]
        rows.append({
            "Periodo": period + 1, "Media": sample.mean(), "Mediana": np.median(sample),
            "Desviación": sample.std(ddof=1), "Mínimo": sample.min(), "Máximo": sample.max(),
            "P05": np.quantile(sample, .05), "P95": np.quantile(sample, .95),
            "Límite central inferior": np.quantile(sample, alpha / 2),
            "Límite central superior": np.quantile(sample, 1 - alpha / 2),
        })
    return pd.DataFrame(rows)
