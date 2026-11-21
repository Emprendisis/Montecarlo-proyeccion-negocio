# Monte Carlo Multiperiodo · Versión 2

Tablero en Streamlit orientado a pronóstico probabilístico y agregación de eventos.

## Inicio y publicación

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para publicar: cargar **todos** los archivos en la raíz de un repositorio de GitHub y desplegar desde https://share.streamlit.io con archivo principal `app.py`.

## Módulos incluidos

1. Importación CSV/Excel y selección de columnas de fecha, valor por evento y eventos por mes.
2. Proyección de 1 a 120 meses; 100 a 100,000 trayectorias (límite global de 8 millones de eventos para proteger memoria).
3. Distribuciones de valor: normal, lognormal, gamma, exponencial, uniforme, triangular y empírica.
4. Distribuciones de número de eventos: Poisson, binomial, fija y empírica.
5. Estimación de parámetros desde historial: media/desviación, media de logs, Gamma por máxima verosimilitud, media exponencial, límites, mediana triangular aproximada; frecuencia Poisson mediante media histórica. La binomial se configura usando oportunidades y probabilidad.
6. Comparación de distribuciones candidatas con AIC, no prueba formal de bondad de ajuste.
7. Tendencia exponencial estimada sobre valores agregados mensuales históricos (mínimo 6 meses) o tasa manual. Estacionalidad multiplicativa estimada por mes calendario (mínimo 12 meses); aplicada al valor, no a la ocurrencia.
8. Trayectorias, bandas percentílicas, histogramas, curva S, estadísticos, probabilidades de excedencia y Excel descargable.

## Significado de los periodos

Cada iteración es una trayectoria de `H` meses futuros. Por ejemplo, `3,000` trayectorias y `12` meses generan 36,000 observaciones mensuales, además de 3,000 totales anuales. La frecuencia de eventos debe estar expresada **por mes**. No debe ingresarse frecuencia anual sin convertir la unidad.

## Qué hace la base histórica

- El historial de severidad alimenta ajuste paramétrico opcional o remuestreo empírico.
- El historial de frecuencia alimenta estimación de λ para Poisson o remuestreo.
- La fecha y valores permiten tendencia y estacionalidad; las filas de un mismo mes se agregan antes de estimarlas.
- La comparación AIC ayuda a seleccionar una familia para valores continuos, pero no certifica adecuación predictiva. La decisión es del analista.

## Interpretación y precauciones

- Resultado agregado por periodo: `S_t = sum(X_{t,i}, i=1..N_t)`, es decir, simulación de distribución compuesta por suma de severidades, no producto de muestras.
- La tendencia y estacionalidad multiplican las severidades agregadas y no modifican `N_t`.
- Las frecuencias y las severidades se consideran independientes. Las observaciones de periodos distintos también son independientes condicionalmente al modelo; no hay autocorrelación simulada.
- La correlación histórica Spearman se informa como **diagnóstico**, no se impone a las muestras simuladas.
- El intervalo central es un rango de percentiles de resultados por periodo. El IC de la media del total cuantifica error de Monte Carlo, condicionado a parámetros fijos, y no incertidumbre de estimación.
- Los agregados de valores de periodos distintos se suman sin descuento financiero; si se usa para VAN, hace falta incorporar curva de tasas y descuento.
- Los ejemplos son sintéticos, para prueba funcional, y no representan pronósticos comerciales.
