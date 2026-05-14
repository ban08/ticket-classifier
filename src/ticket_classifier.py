"""Core helpers for the Sistrade ticket classification project."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import unicodedata

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_sistrade_tickets.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"

RANDOM_STATE = 42

PREDICTION_LABELS = ["ticket_type", "priority", "target_team", "module", "component"]

OUTPUT_LABELS = {
    "ticket_type": "correct_ticket_type",
    "priority": "correct_priority",
    "target_team": "correct_target_team",
    "module": "module",
    "component": "component",
}

EXPECTED_DATA_COLUMNS = [
    "ticket_id",
    "client_name",
    "client_sector",
    "ticket_subject",
    "ticket_description",
    "module",
    "component",
    "urgency_signals",
    "previous_classification",
    "correct_ticket_type",
    "correct_priority",
    "correct_target_team",
    "preferred_resolver",
]

TICKET_FIELDS = [
    "ticket_subject",
    "ticket_description",
    "client_sector",
    "urgency_signals",
    "previous_classification",
    "preferred_resolver",
]

KEYWORD_RULES: dict[str, list[str]] = {
    "bug/error": ["erro", "error", "exception", "stacktrace", "crash", "500", "bug", "nao grava", "duplicado"],
    "user doubt": ["como faco", "onde esta", "duvida", "nao encontro", "manual"],
    "configuration": ["configurar", "parametrizacao", "permissao", "workflow", "template", "perfil", "role"],
    "data issue": ["dados", "importacao", "saldo", "registos", "records", "csv", "excel", "duplicados"],
    "training": ["formacao", "treino", "training", "novos utilizadores", "workshop", "sessao"],
    "infrastructure": ["servidor", "server", "vpn", "rede", "network", "timeout", "backup", "certificado", "lento"],
    "feature request": ["nova opcao", "melhoria", "feature", "seria possivel", "gostavamos"],
    "urgent": ["urgente", "urgent", "asap", "critico", "producao parada", "sem faturar", "blocked", "bloqueado", "hoje"],
    "module CRM": ["crm", "cliente", "contacto", "oportunidade", "pipeline", "comercial"],
    "module invoicing": ["fatura", "factura", "invoice", "e-invoice", "serie", "iva"],
    "module stock": ["stock", "armazem", "barcode", "inventario"],
    "module production": ["producao", "ordem de fabrico", "shop floor", "planeamento"],
    "module accounting": ["contabilidade", "accounting", "saft", "reconciliacao"],
    "module reporting": ["relatorio", "dashboard", "kpi", "excel", "export"],
    "module permissions": ["permissao", "perfil", "role", "acesso", "password"],
    "module integrations": ["api", "edi", "webservice", "integracao", "ecommerce"],
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", text).strip()


def build_ticket_text(ticket: dict[str, Any] | Any) -> str:
    """Build the model input text from realistic ticket fields only."""
    getter = ticket.get if isinstance(ticket, dict) else ticket.__getitem__

    def get(field: str) -> str:
        try:
            value = getter(field)
        except Exception:
            value = ""
        return "" if value is None else str(value)

    parts = [
        ("subject", get("ticket_subject") or get("subject")),
        ("description", get("ticket_description") or get("description")),
        ("client_sector", get("client_sector")),
        ("urgency", get("urgency_signals")),
        ("previous_classification", get("previous_classification")),
        ("preferred_resolver", get("preferred_resolver")),
    ]
    return "\n".join(f"{name}: {value}" for name, value in parts if value)


def contains_keyword(text: str, keyword: str) -> bool:
    clean_text = normalize_text(text)
    clean_keyword = normalize_text(keyword)
    if not clean_keyword:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(clean_keyword)}(?![a-z0-9])"
    return re.search(pattern, clean_text) is not None


def explain_ticket(ticket: dict[str, Any], max_items: int = 6) -> list[str]:
    text = "\n".join(str(ticket.get(field, "")) for field in TICKET_FIELDS if ticket.get(field))
    explanations: list[str] = []
    for rule_name, keywords in KEYWORD_RULES.items():
        matches = [keyword for keyword in keywords if contains_keyword(text, keyword)]
        if matches:
            explanations.append(f"{rule_name}: {', '.join(sorted(matches)[:3])}")
        if len(explanations) >= max_items:
            break
    return explanations or ["No strong keyword evidence; prediction is based on learned ticket text patterns."]


def load_model(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python src/train_model.py` first.")
    return joblib.load(path)


def top_scores(pipeline: Any, text: str, limit: int = 4) -> dict[str, float]:
    probabilities = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return {label: round(float(score), 4) for label, score in ranked[:limit]}


def predict_ticket(ticket: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_model()
    text = build_ticket_text(ticket)
    prediction: dict[str, Any] = {}

    for output_name in PREDICTION_LABELS:
        pipeline = artifact["models"][output_name]
        label = str(pipeline.predict([text])[0])
        scores = top_scores(pipeline, text)
        prediction[output_name] = {
            "label": label,
            "confidence": scores.get(label, max(scores.values())),
            "top_scores": scores,
        }

    prediction["explanation"] = explain_ticket(ticket)
    return prediction


def flatten_prediction(prediction: dict[str, Any]) -> dict[str, str]:
    return {key: prediction[key]["label"] for key in PREDICTION_LABELS}
