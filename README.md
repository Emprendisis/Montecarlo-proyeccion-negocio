# Simulador Monte Carlo · Streamlit

Aplicación de tablero para simulación Monte Carlo con entrada CSV/XLSX, distribuciones configurables, histogramas, curva S (CDF), intervalos y exportación.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar con GitHub y Streamlit Community Cloud

1. Cree un repositorio en GitHub y cargue `app.py`, `requirements.txt` y opcionalmente `datos_ejemplo.csv` en la raíz.
2. En [share.streamlit.io](https://share.streamlit.io), seleccione **Create app**, el repositorio y la rama.
3. Especifique `app.py` como archivo principal y despliegue.

## Interpretación de las entradas

- **Severidad / valor:** importe asociado a un evento, positivo o negativo según el caso. Por ejemplo, pérdida por siniestro, monto por venta o costo por incidente.
- **Frecuencia:** cantidad no negativa y entera de eventos observados en cada periodo (los periodos deben ser homogéneos: todos meses, todos años, etc.).
- **Una variable por iteración:** una extracción de valor por iteración; no utiliza frecuencia.
- **Frecuencia × severidad agregada:** para cada iteración se sortea el número de eventos y después tantos valores como eventos; el resultado es su suma.
- **Empírica:** remuestrea observaciones de la columna elegida, con reemplazo. En el caso paramétrico, el usuario define manualmente los parámetros; **no** se ajustan automáticamente a la base cargada.

`datos_ejemplo.csv` contiene columnas de severidad y frecuencia para explorar la interfaz. Sus filas son **ilustrativas**, no datos reales de un caso financiero.

## Intervalos estadísticos

- El **intervalo de confianza de la media** usa la distribución t de Student y el error estándar de las simulaciones. Describe la precisión numérica de la media estimada *para los parámetros elegidos*; no incorpora incertidumbre sobre dichos parámetros.
- El **intervalo central de resultados** emplea los cuantiles `(1 - confianza)/2` y `(1 + confianza)/2` y describe resultados por periodo.
- La **curva S** es una función de distribución acumulada empírica (CDF): `P(Resultado ≤ x)`.
- La **función de frecuencia** es la distribución del número de eventos por periodo (Poisson, binomial, fija o empírica), distinta del histograma de resultados.

**Advertencia:** el supuesto de independencia entre frecuencia y severidad es explícito. Para pronósticos financieros con correlaciones, estacionalidad o dependencias intertemporales, el modelo requiere extensiones.
