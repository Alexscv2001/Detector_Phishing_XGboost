from __future__ import annotations
import ipaddress
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_xgboost_optimizado.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


app = Flask(__name__)


@dataclass(frozen=True)
class ModelBundle:
    model: object
    scaler: object
    feature_names: list[str]


def load_bundle() -> ModelBundle:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        scaler = joblib.load(SCALER_PATH)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if not feature_names:
        feature_names = [f"feature_{index}" for index in range(getattr(scaler, "n_features_in_", 30))]
    return ModelBundle(model=model, scaler=scaler, feature_names=feature_names)


BUNDLE = load_bundle()


def normalize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("Ingresa una URL para analizar.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = f"http://{url}"
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("La URL no tiene un dominio valido.")
    return url


def hostname_without_www(hostname: str) -> str:
    host = (hostname or "").lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def has_ip_address(hostname: str) -> bool:
    host = hostname_without_www(hostname)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", host))


def suspicious_length(url: str) -> int:
    if len(url) < 54:
        return 1
    if len(url) <= 75:
        return 0
    return -1


def subdomain_score(hostname: str) -> int:
    host = hostname_without_www(hostname)
    labels = [part for part in host.split(".") if part]
    subdomains = max(len(labels) - 2, 0)
    if subdomains == 0:
        return 1
    if subdomains == 1:
        return 0
    return -1


def ssl_score(parsed) -> int:
    if parsed.scheme == "https":
        return 1
    return -1


def url_has_shortener(hostname: str) -> bool:
    shorteners = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "s.id",
        "shorturl.at",
        "rb.gy",
        "lnkd.in",
    }
    host = hostname_without_www(hostname)
    return host in shorteners or any(host.endswith(f".{domain}") for domain in shorteners)


def contains_brand_impersonation(hostname: str) -> bool:
    host = hostname_without_www(hostname)
    brands = ("paypal", "google", "microsoft", "apple", "amazon", "facebook", "netflix", "bank", "bcp", "bbva")
    return any(brand in host and not host.endswith(f"{brand}.com") for brand in brands)


def extract_features(raw_url: str) -> tuple[str, dict[str, int], list[dict[str, object]]]:
    url = normalize_url(raw_url)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    netloc = parsed.netloc or ""
    path_and_query = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    host = hostname_without_www(hostname)

    ip_used = has_ip_address(hostname)
    shortener = url_has_shortener(hostname)
    has_at = "@" in netloc or "@" in path_and_query
    double_slash = "//" in url[url.find("://") + 3 :]
    hyphen_domain = "-" in host
    https_token = "https" in host
    nonstandard_port = parsed.port not in (None, 80, 443)
    abnormal_url = bool(host and host not in url.lower())
    redirect_count = len(re.findall(r"//", url)) - 1
    suspicious_words = re.search(r"(login|verify|secure|account|update|confirm|signin|webscr|wallet)", url, re.I)

    features = {
        "having_IPhaving_IP_Address": -1 if ip_used else 1,
        "URLURL_Length": suspicious_length(url),
        "Shortining_Service": -1 if shortener else 1,
        "having_At_Symbol": -1 if has_at else 1,
        "double_slash_redirecting": -1 if double_slash else 1,
        "Prefix_Suffix": -1 if hyphen_domain else 1,
        "having_Sub_Domain": subdomain_score(hostname),
        "SSLfinal_State": ssl_score(parsed),
        "Domain_registeration_length": 0,
        "Favicon": 0,
        "port": -1 if nonstandard_port else 1,
        "HTTPS_token": -1 if https_token else 1,
        "Request_URL": 0,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 0,
        "Submitting_to_email": -1 if "mailto:" in url.lower() else 1,
        "Abnormal_URL": -1 if abnormal_url else 1,
        "Redirect": -1 if redirect_count > 1 else 1,
        "on_mouseover": 0,
        "RightClick": 0,
        "popUpWidnow": 0,
        "Iframe": 0,
        "age_of_domain": 0,
        "DNSRecord": -1 if ip_used else 0,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": -1 if suspicious_words or contains_brand_impersonation(hostname) else 1,
    }

    explanations = [
        {
            "name": "IP en el dominio",
            "value": features["having_IPhaving_IP_Address"],
            "detail": "Usa una direccion IP directa." if ip_used else "No usa una direccion IP directa.",
        },
        {"name": "Longitud de URL", "value": features["URLURL_Length"], "detail": f"{len(url)} caracteres."},
        {
            "name": "Acortador",
            "value": features["Shortining_Service"],
            "detail": "Dominio asociado a servicios de URL corta." if shortener else "No coincide con acortadores conocidos.",
        },
        {"name": "HTTPS", "value": features["SSLfinal_State"], "detail": f"Esquema detectado: {parsed.scheme}."},
        {"name": "Subdominios", "value": features["having_Sub_Domain"], "detail": f"Host analizado: {host or hostname}."},
        {"name": "Puerto", "value": features["port"], "detail": "Puerto no estandar detectado." if nonstandard_port else "Puerto web estandar o no especificado."},
        {"name": "Palabras sensibles", "value": features["Statistical_report"], "detail": "Busca terminos y marcas usados en suplantacion."},
    ]

    ordered_features = {name: int(features.get(name, 0)) for name in BUNDLE.feature_names}
    return url, ordered_features, explanations


def predict_url(url: str) -> dict[str, object]:
    normalized_url, features, explanations = extract_features(url)
    frame = pd.DataFrame([features], columns=BUNDLE.feature_names)
    scaled = BUNDLE.scaler.transform(frame)
    probabilities = BUNDLE.model.predict_proba(scaled)[0]
    prediction = int(BUNDLE.model.predict(scaled)[0])

    legitimate_probability = float(probabilities[1])
    phishing_probability = float(probabilities[0])
    risk_score = round(phishing_probability * 100, 2)
    label = "Legitimo" if prediction == 1 else "Phishing"
    status = "safe" if prediction == 1 else "danger"

    return {
        "url": normalized_url,
        "label": label,
        "status": status,
        "prediction": prediction,
        "risk_score": risk_score,
        "confidence": round(max(legitimate_probability, phishing_probability) * 100, 2),
        "probabilities": {
            "phishing": round(phishing_probability * 100, 2),
            "legitimate": round(legitimate_probability * 100, 2),
        },
        "features": features,
        "explanations": explanations,
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "features": len(BUNDLE.feature_names)})


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_url(payload.get("url", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"No se pudo generar la prediccion: {exc}"}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
