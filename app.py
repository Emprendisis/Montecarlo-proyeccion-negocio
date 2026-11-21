"""Tablero financiero Monte Carlo multiperiodo."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from model import (CONTINUOUS, FREQUENCY, estimate_parameters, fit_severity,
                   frequency_draw, historical_trend, numeric_history, project,
                   result_statistics, seasonality_index)


st.set_page_config(page_title="Monte Carlo Multiperiodo", layout="wide", page_icon="📈")
NAVY, ORANGE = "#17365D", "#E97725"
st.markdown("""<style>.block-container{max-width:1600px;padding-top:1.6rem}
h1,h2,h3{color:#17365D}div[data-testid="stMetric"]{background:#F3F6FA;
border:1px solid #DDE5EE;border-radius:10px;padding:12px}</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_uploaded(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(BytesIO(raw))
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(BytesIO(raw), encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError):
            pass
    raise ValueError("No fue posible leer el CSV.")


def select_column(label: str, cols: list[str], default: str | None = None) -> str | None:
    values = ["(ninguna)"] + cols
    index = values.index(default) if default in values else 0
    choice = st.selectbox(label, values, index=index)
    return None if choice == "(ninguna)" else choice


def numeric(label: str, default: float, minimum: float | None = None) -> float:
    kwargs = {"value": float(default), "step": 0.1}
    if minimum is not None:
        kwargs["min_value"] = float(minimum)
    return float(st.number_input(label, **kwargs))


def severity_controls(kind: str, estimates: dict) -> dict:
    def v(name, default, min_value=None):
        return numeric(name, estimates.get(name, default), min_value)
    if kind == "Normal":
        return {"media": v("media", 1000), "desv": v("desv", 200, 1e-6)}
    if kind == "Lognormal":
        st.caption("μ y σ corresponden a ln(X).")
        return {"mu_log": v("mu_log", 6.8), "sigma_log": v("sigma_log", .3, 1e-6)}
    if kind == "Gamma":
        return {"forma": v("forma", 3, 1e-6), "escala": v("escala", 300, 1e-6)}
    if kind == "Exponencial":
        return {"escala": v("escala", 1000, 1e-6)}
    if kind == "Uniforme":
        return {"min": v("min", 500), "max": v("max", 1500)}
    if kind == "Triangular":
        return {"min": v("min", 500), "moda": v("moda", 1000), "max": v("max", 1500)}
    return {}


def export_excel(stats_df: pd.DataFrame, results: np.ndarray, freq: np.ndarray,
                 config: dict, start: pd.Timestamp) -> bytes:
    out = BytesIO()
    labels = [str(d.date()) for d in pd.date_range(start, periods=results.shape[1], freq="MS")]
    values_df = pd.DataFrame(results, columns=labels)
    values_df.insert(0, "Escenario", np.arange(1, len(values_df) + 1))
    freq_df = pd.DataFrame(freq, columns=labels)
    freq_df.insert(0, "Escenario", np.arange(1, len(freq_df) + 1))
    cumulative = pd.DataFrame({"Escenario": values_df["Escenario"], "Total horizonte": results.sum(axis=1)})
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Estadisticos", index=False)
        values_df.to_excel(writer, sheet_name="Trayectorias", index=False)
        freq_df.to_excel(writer, sheet_name="Eventos", index=False)
        cumulative.to_excel(writer, sheet_name="Total horizonte", index=False)
        pd.DataFrame([(k, str(v)) for k, v in config.items()], columns=["Configuración", "Valor"]).to_excel(writer, sheet_name="Parametros", index=False)
    return out.getvalue()


st.title("Simulador Monte Carlo · Multiperiodo")
st.caption("Proyección de escenarios financieros y riesgo compuesto · distribución, frecuencia, tendencia y estacionalidad")

with st.sidebar:
    st.header("1. Base histórica")
    file = st.file_uploader("CSV o Excel", type=["csv", "xlsx"])
    frame = None
    if file:
        try:
            frame = read_uploaded(file.getvalue(), file.name)
            st.success(f"{len(frame):,} registros importados")
        except Exception as exc:
            st.error(str(exc))
    cols = list(frame.columns) if frame is not None else []
    sev_col = select_column("Severidad / valor por evento", cols, "Severidad")
    freq_col = select_column("Eventos por periodo", cols, "Eventos")
    date_col = select_column("Fecha de observación", cols, "Fecha")
    try:
        sev_hist = numeric_history(frame, sev_col) if frame is not None else None
        freq_hist = numeric_history(frame, freq_col, counts=True) if frame is not None else None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.header("2. Escenarios")
    paths = st.number_input("Número de trayectorias", min_value=100, max_value=100_000, value=3000, step=100)
    horizon = st.number_input("Horizonte (meses)", min_value=1, max_value=120, value=12)
    start_date = st.date_input("Primer mes proyectado", value=pd.Timestamp("2027-01-01").date())
    start = pd.Timestamp(start_date).to_period("M").to_timestamp()
    conf_pct = st.slider("Nivel de confianza (%)", 80, 99, 95)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=2_147_483_647, value=2026)
    model_type = st.radio("Motor de simulación", ["Variable simple", "Frecuencia × severidad"], index=1)
    aggregated = model_type == "Frecuencia × severidad"

    st.header("3. Distribución de probabilidad")
    sev_dist = st.selectbox("Valor por evento", CONTINUOUS)
    automatic = st.checkbox("Estimar parámetros desde la base", value=False, disabled=sev_hist is None or sev_dist == "Empírica")
    estimates = {}
    if automatic and sev_hist is not None:
        try:
            estimates = estimate_parameters(sev_hist, sev_dist)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    if sev_dist == "Empírica" and sev_hist is None:
        st.warning("La distribución empírica requiere una columna histórica seleccionada.")
    sev_params = severity_controls(sev_dist, estimates)

    freq_dist = "Fija"
    freq_params = {"eventos": 1}
    if aggregated:
        st.header("4. Distribución de frecuencia")
        freq_dist = st.selectbox("Eventos por mes", FREQUENCY)
        avg_events = float(freq_hist.mean()) if freq_hist is not None else 5.
        if freq_dist == "Poisson":
            default = avg_events if st.checkbox("Estimar λ de la frecuencia histórica", value=freq_hist is not None, disabled=freq_hist is None) else 5.
            freq_params = {"lambda": numeric("λ eventos/mes", default, 0)}
        elif freq_dist == "Binomial":
            trials = st.number_input("Oportunidades/mes", min_value=1, max_value=100000, value=10)
            p_init = min(avg_events / trials, 1.) if freq_hist is not None else .4
            freq_params = {"ensayos": int(trials), "p": st.slider("Probabilidad por oportunidad", 0., 1., float(p_init), .01)}
        elif freq_dist == "Fija":
            freq_params = {"eventos": st.number_input("Eventos fijos/mes", min_value=0, max_value=100000, value=5)}
        elif freq_hist is None:
            st.warning("La frecuencia empírica requiere datos históricos.")
            freq_params = {}
        else:
            freq_params = {}

    st.header("5. Dinámica temporal")
    trend_mode = st.selectbox("Tendencia mensual", ["Sin tendencia", "Manual", "Estimar con histórico"])
    growth_pct = 0.
    trend_error = None
    if trend_mode == "Manual":
        growth_pct = numeric("Crecimiento mensual (%)", 0.)
    elif trend_mode == "Estimar con histórico":
        if frame is None or not date_col or not sev_col:
            trend_error = "Para estimar tendencia seleccione fecha y valor histórico."
        else:
            try:
                growth_pct = 100 * historical_trend(frame, date_col, sev_col)
                st.caption(f"Tendencia estimada: {growth_pct:.3f}% mensual")
            except ValueError as exc:
                trend_error = str(exc)
    season_mode = st.checkbox("Aplicar estacionalidad histórica mensual")
    seasonal = None
    season_error = None
    if season_mode:
        if frame is None or not date_col or not sev_col:
            season_error = "La estacionalidad requiere fechas y valores históricos."
        else:
            try:
                seasonal = seasonality_index(frame, date_col, sev_col)
            except ValueError as exc:
                season_error = str(exc)
    if trend_error:
        st.warning(trend_error)
    if season_error:
        st.warning(season_error)
    st.caption("La tendencia y estacionalidad escalan los importes simulados, no la frecuencia de eventos.")
    run = st.button("Ejecutar proyección", type="primary", width="stretch")

if frame is not None:
    with st.expander("Explorar base de datos y diagnóstico estadístico"):
        st.dataframe(frame.head(100), width="stretch", hide_index=True)
        if sev_hist is not None:
            a, b, c = st.columns(3)
            a.metric("Observaciones válidas", f"{len(sev_hist):,}")
            b.metric("Media histórica", f"{sev_hist.mean():,.2f}")
            c.metric("Desviación histórica", f"{sev_hist.std(ddof=1):,.2f}" if len(sev_hist) > 1 else "N/D")
            if len(sev_hist) >= 5 and np.std(sev_hist) > 0:
                try:
                    fit_table = fit_severity(sev_hist)
                    st.markdown("**Comparación de distribuciones continuas por AIC** (menor es preferible entre candidatas)")
                    st.dataframe(fit_table, width="stretch", hide_index=True)
                    st.caption("El AIC no es prueba formal de bondad de ajuste ni reemplaza la revisión de colas y supuestos.")
                except ValueError as exc:
                    st.info(str(exc))
        if sev_col and freq_col and sev_col != freq_col:
            paired = frame[[sev_col, freq_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) >= 5:
                correlation = paired.corr(method="spearman").iloc[0, 1]
                st.caption(f"Asociación descriptiva por fila (Spearman): {correlation:.3f}. No se incorpora automáticamente en las simulaciones.")

if run:
    if trend_error or season_error:
        st.error("Corrija las entradas temporales antes de ejecutar.")
    elif growth_pct <= -100:
        st.error("El crecimiento mensual debe ser superior a -100%.")
    else:
        try:
            with st.spinner("Generando trayectorias..."):
                values, frequencies = project(
                    iterations=int(paths), horizon=int(horizon), seed=int(seed), aggregated=aggregated,
                    sev_dist=sev_dist, sev_params=sev_params, sev_hist=sev_hist,
                    freq_dist=freq_dist, freq_params=freq_params, freq_hist=freq_hist,
                    monthly_growth=growth_pct / 100, seasonal=seasonal, start_month=start.month,
                )
            st.session_state["projection"] = (values, frequencies)
            st.session_state["projection_settings"] = {
                "Inicio": str(start.date()), "Escenarios": int(paths), "Meses": int(horizon),
                "Nivel de confianza": f"{conf_pct}%", "Semilla": int(seed), "Modelo": model_type,
                "Distribución de valor": sev_dist, "Parámetros de valor": sev_params,
                "Distribución de frecuencia": freq_dist if aggregated else "No aplica",
                "Parámetros de frecuencia": freq_params if aggregated else {},
                "Tendencia mensual": growth_pct / 100, "Índices estacionales": seasonal.tolist() if seasonal is not None else "Sin estacionalidad",
                "Columnas históricas": {"valor": sev_col, "frecuencia": freq_col, "fecha": date_col},
            }
            st.session_state["projection_confidence"] = conf_pct / 100
        except (ValueError, OverflowError, MemoryError) as exc:
            st.error(f"No se ejecutó el modelo: {exc}")

if "projection" not in st.session_state:
    st.info("Cargue datos opcionalmente, seleccione las distribuciones y pulse «Ejecutar proyección».")
    st.stop()

values, frequencies = st.session_state["projection"]
config = st.session_state["projection_settings"]
confidence = st.session_state["projection_confidence"]
stats_df = result_statistics(values, confidence)
total = values.sum(axis=1)
dates = pd.date_range(pd.Timestamp(config["Inicio"]), periods=values.shape[1], freq="MS")
alpha = 1 - confidence
overall_low, overall_high = np.quantile(total, [alpha / 2, 1 - alpha / 2])

st.subheader("Indicadores del horizonte completo")
a, b, c, d, e = st.columns(5)
a.metric("Total esperado", f"{total.mean():,.2f}")
b.metric("Mediana total", f"{np.median(total):,.2f}")
c.metric("Desv. total", f"{total.std(ddof=1):,.2f}")
d.metric("Percentil 95", f"{np.quantile(total, .95):,.2f}")
e.metric("Horizonte", f"{values.shape[1]} meses")
st.caption(f"Intervalo central del total ({confidence:.0%}): {overall_low:,.2f} a {overall_high:,.2f}. Valores sin descuento financiero.")

t1, t2, t3, t4 = st.tabs(["Trayectorias y bandas", "Histogramas y curva S", "Estadísticos", "Exportación y supuestos"])
with t1:
    plot = go.Figure()
    hi = stats_df["Límite central superior"].to_numpy()
    lo = stats_df["Límite central inferior"].to_numpy()
    plot.add_trace(go.Scatter(x=dates, y=hi, line=dict(width=0), hoverinfo="skip", showlegend=False))
    plot.add_trace(go.Scatter(x=dates, y=lo, fill="tonexty", fillcolor="rgba(23,54,93,.17)", line=dict(width=0), name=f"Intervalo central {confidence:.0%}"))
    plot.add_trace(go.Scatter(x=dates, y=stats_df["Media"], line=dict(color=NAVY, width=3), name="Media"))
    plot.add_trace(go.Scatter(x=dates, y=stats_df["Mediana"], line=dict(color=ORANGE, width=2, dash="dash"), name="Mediana"))
    for idx in np.linspace(0, len(values) - 1, min(len(values), 20), dtype=int):
        plot.add_trace(go.Scatter(x=dates, y=values[idx], line=dict(color="rgba(130,140,150,.18)", width=1), showlegend=False, hoverinfo="skip"))
    plot.update_layout(title="Evolución mensual · escenarios y percentiles", xaxis_title="Mes", yaxis_title="Resultado por mes", template="plotly_white", hovermode="x unified")
    st.plotly_chart(plot, width="stretch")
    st.caption("Las líneas grises representan una muestra de escenarios. El intervalo central describe resultados posibles por mes, no la confianza de la media.")
with t2:
    selection = st.selectbox("Distribución a analizar", ["Total del horizonte"] + [f"Mes {i}" for i in range(1, values.shape[1] + 1)])
    sample = total if selection == "Total del horizonte" else values[:, int(selection.split()[1]) - 1]
    left, right = st.columns(2)
    with left:
        bins = st.slider("Barras del histograma", 10, 100, 40)
        hist = go.Figure(go.Histogram(x=sample, nbinsx=bins, histnorm="probability density", marker_color=NAVY))
        hist.add_vline(x=float(np.mean(sample)), line_color=ORANGE, annotation_text="Media")
        hist.update_layout(title=f"Histograma · {selection}", xaxis_title="Resultado", yaxis_title="Densidad", template="plotly_white")
        st.plotly_chart(hist, width="stretch")
    with right:
        ordered = np.sort(sample)
        ecdf = go.Figure(go.Scatter(x=ordered, y=np.arange(1, len(sample) + 1) / len(sample), mode="lines", line=dict(color=ORANGE, width=3)))
        ecdf.update_layout(title="Curva S · probabilidad acumulada", xaxis_title="Resultado ≤ x", yaxis_title="Probabilidad acumulada", yaxis_range=[0, 1], template="plotly_white")
        st.plotly_chart(ecdf, width="stretch")
    threshold = st.number_input("Umbral de excedencia", value=float(np.median(sample)))
    l, r = st.columns(2)
    l.metric("P(Resultado ≤ umbral)", f"{np.mean(sample <= threshold):.2%}")
    r.metric("P(Resultado > umbral)", f"{np.mean(sample > threshold):.2%}")
    if config["Modelo"] == "Frecuencia × severidad":
        st.markdown("**Frecuencia total por escenario**")
        freq_total = frequencies.sum(axis=1)
        f = go.Figure(go.Histogram(x=freq_total, marker_color=NAVY))
        f.update_layout(xaxis_title="Eventos en el horizonte", yaxis_title="Escenarios", template="plotly_white")
        st.plotly_chart(f, width="stretch")
with t3:
    table = stats_df.copy()
    table.insert(1, "Mes", dates.strftime("%Y-%m"))
    st.dataframe(table.style.format(precision=3, thousands=","), width="stretch", hide_index=True)
    standard_error = float(stats.sem(total))
    margin = float(stats.t.ppf((1 + confidence) / 2, len(total) - 1) * standard_error)
    st.markdown(f"**IC t aproximado para la media del total ({confidence:.0%}):** [{total.mean()-margin:,.2f}, {total.mean()+margin:,.2f}]")
    st.caption("Este IC cuantifica error de simulación, condicionado al modelo; no incluye incertidumbre de parámetros o error de especificación.")
with t4:
    st.dataframe(pd.DataFrame([(k, str(v)) for k, v in config.items()], columns=["Configuración", "Valor"]), width="stretch", hide_index=True)
    st.download_button("Descargar Excel: estadísticas y todas las trayectorias", export_excel(stats_df, values, frequencies, config, dates[0]),
                       "MonteCarlo_Multiperiodo.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    st.download_button("Descargar CSV de estadísticos mensuales", stats_df.to_csv(index=False).encode("utf-8-sig"), "MonteCarlo_Estadisticos.csv", "text/csv")
    st.markdown("""
    **Supuestos y límites:** las simulaciones por periodo son independientes, condicionales a los parámetros; severidad y frecuencia también son independientes. La tendencia y la estacionalidad se aplican al importe, no al conteo. El ajuste AIC compara familias candidatas, pero no garantiza validez predictiva. La estimación automática no incorpora intervalos de incertidumbre de parámetros, correlaciones entre variables ni procesos autorregresivos. Las sumas no se descuentan a valor presente.
    """)
