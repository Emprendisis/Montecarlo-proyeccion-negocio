"""Tablero Monte Carlo: severidad, frecuencia y riesgo agregado."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st


st.set_page_config(page_title="Simulador Monte Carlo", page_icon="📊", layout="wide")

BLUE = "#17365D"
ORANGE = "#E97725"
st.markdown(
    """<style>
    .block-container {padding-top:1.5rem;max-width:1500px;}
    h1,h2,h3 {color:#17365D;}
    div[data-testid="stMetric"] {background:#F3F6FA;border:1px solid #DDE5EE;
       border-radius:12px;padding:14px;}
    </style>""",
    unsafe_allow_html=True,
)


def load_data(uploaded) -> pd.DataFrame:
    """CSV UTF-8/Latin-1 o XLSX, sin modificar la fuente."""
    if uploaded.name.lower().endswith(".xlsx"):
        return pd.read_excel(uploaded)
    raw = uploaded.getvalue()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(BytesIO(raw), encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("No fue posible interpretar el archivo CSV.")


def valid_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    numbers = pd.to_numeric(frame[column], errors="coerce")
    numbers = numbers.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(numbers) == 0:
        raise ValueError(f"La columna '{column}' no contiene observaciones numéricas válidas.")
    return numbers


def sample_severity(rng: np.random.Generator, kind: str, p: dict, size: int, hist: np.ndarray | None) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=float)
    if kind == "Empírica (remuestreo)":
        if hist is None or len(hist) == 0:
            raise ValueError("Se requieren datos históricos para la distribución empírica.")
        return rng.choice(hist, size=size, replace=True)
    if kind == "Normal":
        return rng.normal(p["media"], p["desviacion"], size)
    if kind == "Lognormal":
        return rng.lognormal(p["mu_log"], p["sigma_log"], size)
    if kind == "Triangular":
        return rng.triangular(p["minimo"], p["moda"], p["maximo"], size)
    if kind == "Uniforme":
        return rng.uniform(p["minimo"], p["maximo"], size)
    if kind == "Exponencial":
        return rng.exponential(p["escala"], size)
    raise ValueError("Distribución de severidad desconocida")


def sample_frequency(rng: np.random.Generator, kind: str, p: dict, n: int, hist: np.ndarray | None) -> np.ndarray:
    if kind == "Fija":
        return np.full(n, p["eventos"], dtype=np.int64)
    if kind == "Poisson":
        return rng.poisson(p["lambda"], n)
    if kind == "Binomial":
        return rng.binomial(p["ensayos"], p["probabilidad"], n)
    if kind == "Empírica (remuestreo)":
        if hist is None or len(hist) == 0:
            raise ValueError("Se requieren frecuencias históricas para el remuestreo.")
        if np.any(hist < 0) or np.any(hist != np.floor(hist)):
            raise ValueError("La frecuencia histórica debe contener enteros no negativos.")
        return rng.choice(hist.astype(np.int64), size=n, replace=True)
    raise ValueError("Distribución de frecuencia desconocida")


def simulate(n: int, seed: int, mode: str, severity_kind: str, severity_params: dict,
             severity_history: np.ndarray | None, frequency_kind: str, frequency_params: dict,
             frequency_history: np.ndarray | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if mode == "Una variable por iteración":
        values = sample_severity(rng, severity_kind, severity_params, n, severity_history)
        counts = np.ones(n, dtype=np.int64)
    else:
        counts = sample_frequency(rng, frequency_kind, frequency_params, n, frequency_history)
        if np.any(counts < 0):
            raise ValueError("Las frecuencias no pueden ser negativas.")
        total = int(np.sum(counts, dtype=np.int64))
        if total > 8_000_000:
            raise ValueError("La configuración produciría más de 8 millones de eventos. Reduzca iteraciones o frecuencia.")
        draws = sample_severity(rng, severity_kind, severity_params, total, severity_history)
        # Reducción vectorizada, compatible también con frecuencias cero.
        values = np.zeros(n, dtype=float)
        if total:
            np.add.at(values, np.repeat(np.arange(n), counts), draws)
    if not np.all(np.isfinite(values)):
        raise ValueError("La simulación produjo valores no finitos. Revise los parámetros.")
    return pd.DataFrame({"Iteración": np.arange(1, n + 1), "Eventos": counts, "Resultado": values})


def summary_metrics(values: np.ndarray, confidence: float) -> tuple[pd.DataFrame, float, float]:
    n = len(values)
    avg = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    qlo, qhi = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    if n > 1:
        margin = float(stats.t.ppf((1 + confidence) / 2, df=n - 1) * std / np.sqrt(n))
    else:
        margin = 0.0
    rows = [
        ("Iteraciones", n), ("Media", avg), ("Mediana", float(np.median(values))),
        ("Desviación estándar", std), ("Varianza muestral", std ** 2),
        ("Mínimo", float(np.min(values))), ("Máximo", float(np.max(values))),
        ("Percentil 5", float(np.quantile(values, .05))),
        ("Percentil 25", float(np.quantile(values, .25))),
        ("Percentil 75", float(np.quantile(values, .75))),
        ("Percentil 95", float(np.quantile(values, .95))),
        ("Asimetría", float(stats.skew(values)) if std > 0 else 0.),
        ("Curtosis (exceso)", float(stats.kurtosis(values)) if std > 0 else 0.),
        ("Límite inferior IC media", avg - margin),
        ("Límite superior IC media", avg + margin),
        ("Límite inferior intervalo central", float(qlo)),
        ("Límite superior intervalo central", float(qhi)),
    ]
    return pd.DataFrame(rows, columns=["Estadístico", "Valor"]), avg - margin, avg + margin


def excel_bytes(results: pd.DataFrame, metrics: pd.DataFrame, config: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Estadísticos", index=False)
        results.to_excel(writer, sheet_name="Simulaciones", index=False)
        config.to_excel(writer, sheet_name="Configuración", index=False)
    return out.getvalue()


def numeric_input(label: str, value: float, lower: float | None = None, step: float = 0.1, key: str | None = None) -> float:
    kwargs = {"label": label, "value": float(value), "step": float(step), "key": key}
    if lower is not None:
        kwargs["min_value"] = float(lower)
    return float(st.number_input(**kwargs))


st.title("Simulador Monte Carlo")
st.caption("Tablero de simulación probabilística · distribución de valores y frecuencia de ocurrencia")

with st.sidebar:
    st.header("1. Datos de entrada")
    uploaded = st.file_uploader("Cargar base histórica", type=["csv", "xlsx"])
    if uploaded:
        try:
            df = load_data(uploaded)
            st.success(f"{len(df):,} registros · {len(df.columns)} columnas")
        except Exception as exc:
            st.error(f"Error de lectura: {exc}")
            df = None
    else:
        df = None

    available = list(df.columns) if df is not None else []
    severity_col = st.selectbox("Columna: valor o severidad", ["Sin seleccionar"] + available)
    frequency_col = st.selectbox("Columna: número de eventos por periodo", ["Sin seleccionar"] + available,
                                 help="Cada fila debe representar un periodo comparable (p. ej., mes).")

    st.header("2. Configuración")
    n = st.number_input("Número de iteraciones", min_value=100, max_value=200_000, value=10_000, step=1_000)
    confidence_pct = st.slider("Nivel de confianza (%)", min_value=80, max_value=99, value=95)
    seed = st.number_input("Semilla aleatoria (reproducibilidad)", min_value=0, max_value=2_147_483_647, value=2026)
    mode = st.radio("Tipo de modelo", ["Una variable por iteración", "Frecuencia × severidad agregada"],
                    index=1, help="Agregado: suma de los valores de todos los eventos en cada periodo simulado.")

    st.header("3. Función de probabilidad")
    severity_kind = st.selectbox("Distribución del valor por evento", ["Normal", "Lognormal", "Triangular", "Uniforme", "Exponencial", "Empírica (remuestreo)"])
    sev_params: dict = {}
    if severity_kind == "Normal":
        sev_params["media"] = numeric_input("Media", 1000)
        sev_params["desviacion"] = numeric_input("Desviación estándar", 200, 0.000001)
    elif severity_kind == "Lognormal":
        st.caption("Parámetros de ln(X), no media/desviación de X.")
        sev_params["mu_log"] = numeric_input("Media de ln(X)", 6.8)
        sev_params["sigma_log"] = numeric_input("Desviación de ln(X)", 0.3, 0.000001)
    elif severity_kind == "Triangular":
        sev_params["minimo"] = numeric_input("Mínimo", 500)
        sev_params["moda"] = numeric_input("Moda", 1000)
        sev_params["maximo"] = numeric_input("Máximo", 1800)
    elif severity_kind == "Uniforme":
        sev_params["minimo"] = numeric_input("Mínimo", 500)
        sev_params["maximo"] = numeric_input("Máximo", 1500)
    elif severity_kind == "Exponencial":
        sev_params["escala"] = numeric_input("Escala = media (1/tasa)", 1000, 0.000001)
    else:
        st.info("Utiliza la columna histórica de valor seleccionada.")

    frequency_kind = "Fija"
    freq_params: dict = {}
    if mode == "Frecuencia × severidad agregada":
        st.header("4. Función de frecuencia")
        frequency_kind = st.selectbox("Distribución del número de eventos", ["Poisson", "Binomial", "Fija", "Empírica (remuestreo)"])
        if frequency_kind == "Poisson":
            freq_params["lambda"] = numeric_input("λ: eventos esperados por periodo", 5, 0)
        elif frequency_kind == "Binomial":
            freq_params["ensayos"] = st.number_input("Número de oportunidades por periodo", min_value=0, max_value=100_000, value=10)
            freq_params["probabilidad"] = st.slider("Probabilidad de ocurrencia por oportunidad", 0., 1., .4, .01)
        elif frequency_kind == "Fija":
            freq_params["eventos"] = st.number_input("Eventos por periodo", min_value=0, max_value=100_000, value=5)
        else:
            st.info("Utiliza la columna histórica de frecuencia seleccionada.")

    run = st.button("Ejecutar simulación", type="primary", width="stretch")


with st.expander("Definiciones del modelo y uso de la base histórica", expanded=False):
    st.markdown("""
    - **Probabilidad (severidad):** distribución del importe, pérdida, ingreso o valor de *cada evento*.
    - **Frecuencia:** distribución del *número de eventos por periodo*; no equivale al histograma observado.
    - **Resultado agregado:** suma de los valores por evento en cada iteración/periodo simulado.
    - **Distribución empírica:** selecciona observaciones históricas con reemplazo. Las distribuciones paramétricas se configuran manualmente; la carga de datos **no ajusta automáticamente** sus parámetros.
    - **Intervalo de confianza de la media:** intervalo t aproximado para el valor esperado estimado, condicionado al modelo y a sus parámetros.
    - **Intervalo central de resultados:** percentiles de la distribución simulada, que representan la incertidumbre de *un periodo*, no de la media.
    - **Curva S:** función de distribución acumulada empírica (probabilidad de que el resultado sea menor o igual a un umbral).
    """)

if df is not None:
    with st.expander("Vista previa de la base histórica"):
        st.dataframe(df.head(30), width="stretch")

if run:
    try:
        severity_hist = valid_series(df, severity_col) if df is not None and severity_col != "Sin seleccionar" else None
        frequency_hist = valid_series(df, frequency_col) if df is not None and frequency_col != "Sin seleccionar" else None
        if severity_kind == "Empírica (remuestreo)" and severity_hist is None:
            raise ValueError("Cargue un archivo y seleccione una columna numérica de valor.")
        if mode == "Frecuencia × severidad agregada" and frequency_kind == "Empírica (remuestreo)" and frequency_hist is None:
            raise ValueError("Cargue un archivo y seleccione una columna numérica de frecuencia.")
        if severity_kind == "Triangular" and not (sev_params["minimo"] < sev_params["maximo"] and sev_params["minimo"] <= sev_params["moda"] <= sev_params["maximo"]):
            raise ValueError("Para triangular se requiere mínimo < máximo y mínimo ≤ moda ≤ máximo.")
        if severity_kind == "Uniforme" and sev_params["minimo"] >= sev_params["maximo"]:
            raise ValueError("Para uniforme se requiere mínimo < máximo.")
        result = simulate(int(n), int(seed), mode, severity_kind, sev_params, severity_hist,
                          frequency_kind, freq_params, frequency_hist)
        st.session_state["sim"] = result
        st.session_state["config"] = {
            "Iteraciones": int(n), "Confianza": confidence_pct, "Semilla": int(seed),
            "Modelo": mode, "Distribución de valor": severity_kind,
            "Distribución de frecuencia": frequency_kind if mode != "Una variable por iteración" else "No aplica",
            "Parámetros de valor": str(sev_params), "Parámetros de frecuencia": str(freq_params),
            "Columna de valor": severity_col, "Columna de frecuencia": frequency_col,
        }
    except Exception as exc:
        st.error(f"No fue posible ejecutar la simulación: {exc}")

if "sim" not in st.session_state:
    st.info("Configure los parámetros en la barra lateral y seleccione «Ejecutar simulación».")
    st.stop()

results: pd.DataFrame = st.session_state["sim"]
cfg: dict = st.session_state["config"]
confidence = cfg["Confianza"] / 100
values = results["Resultado"].to_numpy()
metrics, mean_ci_low, mean_ci_high = summary_metrics(values, confidence)
central_low, central_high = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])

st.subheader("Indicadores principales")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Media simulada", f"{values.mean():,.2f}")
c2.metric("Mediana", f"{np.median(values):,.2f}")
c3.metric("Desviación estándar", f"{values.std(ddof=1):,.2f}")
c4.metric("Percentil 95", f"{np.quantile(values, .95):,.2f}")
c5.metric("Iteraciones", f"{len(values):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Distribución y curva S", "Estadísticos", "Frecuencia", "Datos y exportación"])
with tab1:
    left, right = st.columns(2)
    with left:
        bins = st.slider("Intervalos del histograma", min_value=10, max_value=120, value=40)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=values, nbinsx=bins, histnorm="probability density", name="Densidad simulada", marker_color=BLUE, opacity=.82))
        fig.add_vline(x=float(values.mean()), line_color=ORANGE, line_width=2, annotation_text="Media")
        fig.update_layout(title="Histograma de resultados", xaxis_title="Resultado por iteración", yaxis_title="Densidad", bargap=.025, template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with right:
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        fig = go.Figure(go.Scatter(x=ordered, y=cumulative, mode="lines", line=dict(color=ORANGE, width=3), name="F(x)"))
        fig.add_hline(y=confidence, line_dash="dash", line_color=BLUE, annotation_text=f"{confidence:.0%}")
        fig.update_layout(title="Curva S · distribución acumulada", xaxis_title="Resultado ≤ x", yaxis_title="Probabilidad acumulada", yaxis_range=[0, 1], template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    st.markdown(f"**IC de la media ({confidence:.0%}):** [{mean_ci_low:,.2f}, {mean_ci_high:,.2f}]  |  **Intervalo central de resultados ({confidence:.0%}):** [{central_low:,.2f}, {central_high:,.2f}]")
    threshold = st.number_input("Umbral para estimar probabilidad de excedencia", value=float(np.median(values)))
    a, b = st.columns(2)
    a.metric("P(Resultado ≤ umbral)", f"{np.mean(values <= threshold):.2%}")
    b.metric("P(Resultado > umbral)", f"{np.mean(values > threshold):.2%}")

with tab2:
    st.dataframe(metrics.style.format({"Valor": "{:,.4f}"}), width="stretch", hide_index=True)
    st.caption("La varianza es muestral y la curtosis está expresada como exceso respecto de la distribución normal.")

with tab3:
    if cfg["Modelo"] == "Una variable por iteración":
        st.info("El modelo de una variable utiliza exactamente un evento por iteración. Active el modelo agregado para simular la frecuencia.")
    else:
        counts = results["Eventos"].to_numpy()
        st.metric("Eventos promedio por periodo", f"{counts.mean():,.2f}")
        freq_table = results["Eventos"].value_counts().sort_index().rename_axis("Eventos").reset_index(name="Ocurrencias")
        freq_table["Frecuencia relativa"] = freq_table["Ocurrencias"] / len(results)
        freq_table["Frecuencia acumulada"] = freq_table["Frecuencia relativa"].cumsum()
        fig = go.Figure(go.Bar(x=freq_table["Eventos"], y=freq_table["Frecuencia relativa"], marker_color=BLUE))
        fig.update_layout(title="Distribución simulada del número de eventos", xaxis_title="Eventos por periodo", yaxis_title="Frecuencia relativa", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(freq_table, width="stretch", hide_index=True)

with tab4:
    st.dataframe(results.head(1000), width="stretch", hide_index=True)
    config_table = pd.DataFrame([(key, str(value)) for key, value in cfg.items()], columns=["Configuración", "Valor"])
    st.dataframe(config_table, width="stretch", hide_index=True)
    st.download_button("Descargar resultados en CSV", results.to_csv(index=False).encode("utf-8-sig"), "simulaciones_montecarlo.csv", "text/csv")
    st.download_button("Descargar tablero de datos en Excel", excel_bytes(results, metrics, config_table),
                       "resultados_montecarlo.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
