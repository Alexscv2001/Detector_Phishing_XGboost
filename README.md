# PhishGuard ML

Aplicacion web profesional para desplegar un modelo XGBoost de deteccion de phishing.

## Ejecutar

```powershell
.\\.venv\\Scripts\\python.exe app.py
```

Abre `http://127.0.0.1:5000`.

## Estructura

- `app.py`: backend Flask, carga del modelo, extraccion de features y endpoint `/api/predict`.
- `templates/index.html`: interfaz principal.
- `static/css/styles.css`: estilos responsive.
- `static/js/app.js`: consumo del endpoint y render del resultado.
- `modelo_xgboost_optimizado.pkl`: clasificador XGBoost.
- `scaler.pkl`: scaler con el orden de las 30 features.

## Nota tecnica

El scaler conserva los nombres de las 30 variables. La app extrae senales desde la URL y asigna valor neutro a variables que normalmente requieren contenido HTML, datos DNS, trafico web o servicios externos.
