"""Shared utilities for the Sistrade ticket classification POC."""

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

KEYWORD_RULES: dict[str, list[str]] = {
    "bug/error evidence": [
        "erro",
        "error",
        "exception",
        "stacktrace",
        "crash",
        "500",
        "bug",
        "nao grava",
        "não grava",
        "duplicado",
    ],
    "client misunderstanding evidence": [
        "como faco",
        "como faço",
        "onde esta",
        "onde está",
        "duvida",
        "dúvida",
        "nao encontro",
        "não encontro",
        "manual",
    ],
    "configuration evidence": [
        "configurar",
        "parametrizacao",
        "parametrização",
        "permissao",
        "permissão",
        "workflow",
        "template",
        "perfil",
    ],
    "data quality evidence": [
        "dados",
        "importacao",
        "importação",
        "saldo",
        "registos",
        "records",
        "csv",
        "excel",
        "duplicados",
    ],
    "training evidence": [
        "formacao",
        "formação",
        "treino",
        "training",
        "novos utilizadores",
        "workshop",
        "sessao",
        "sessão",
    ],
    "infrastructure evidence": [
        "servidor",
        "server",
        "vpn",
        "rede",
        "network",
        "timeout",
        "backup",
        "certificado",
        "lento",
    ],
    "feature request evidence": [
        "nova opcao",
        "nova opção",
        "melhoria",
        "feature",
        "seria possivel",
        "seria possível",
        "gostavamos",
        "gostávamos",
    ],
    "urgent priority evidence": [
        "urgente",
        "urgent",
        "asap",
        "critico",
        "crítico",
        "producao parada",
        "produção parada",
        "sem faturar",
        "cannot invoice",
        "blocked",
        "bloqueado",
        "hoje",
    ],
    "module: CRM": ["crm", "cliente", "contacto", "oportunidade", "pipeline", "comercial"],
    "module: invoicing": ["fatura", "factura", "invoice", "e-invoice", "serie", "série", "iva"],
    "module: stock": ["stock", "armazem", "armazém", "barcode", "inventario", "inventário"],
    "module: production": ["producao", "produção", "ordem de fabrico", "shop floor", "planeamento"],
    "module: accounting": ["contabilidade", "accounting", "saft", "saft", "reconciliacao", "reconciliação"],
    "module: reporting": ["relatorio", "relatório", "dashboard", "kpi", "excel", "export"],
    "module: permissions": ["permissao", "permissão", "perfil", "role", "acesso", "password"],
    "module: integrations": ["api", "edi", "webservice", "integracao", "integração", "ecommerce"],
}


def normalize_text(value: Any) -> str:
    """Return a compact lowercase string for deterministic keyword matching."""
    if value is None:
        return ""
    text = str(value).lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_ticket_text(ticket: dict[str, Any] | Any) -> str:
    """Build the text feature used by both training and inference.

    This deliberately excludes the target labels (`module`, `component`, and
    correct classifications) so evaluation is not inflated by label leakage.
    """
    getter = ticket.get if isinstance(ticket, dict) else ticket.__getitem__

    def get(field: str) -> str:
        try:
            value = getter(field)
        except Exception:
            value = ""
        if value is None:
            return ""
        return str(value)

    parts = [
        ("subject", get("ticket_subject") or get("subject")),
        ("description", get("ticket_description") or get("description")),
        ("client_sector", get("client_sector")),
        ("urgency_signals", get("urgency_signals")),
        ("previous_classification", get("previous_classification")),
        ("preferred_resolver", get("preferred_resolver")),
    ]
    return "\n".join(f"{name}: {value}" for name, value in parts if value)


def build_explanation_text(ticket: dict[str, Any]) -> str:
    """Build explanation text from user evidence, excluding old classifications."""
    return "\n".join(
        str(ticket.get(field, ""))
        for field in [
            "ticket_subject",
            "subject",
            "ticket_description",
            "description",
            "client_sector",
            "urgency_signals",
            "preferred_resolver",
        ]
        if ticket.get(field)
    )


def contains_keyword(text: str, keyword: str) -> bool:
    """Match keywords with token boundaries to avoid substring false positives."""
    clean_text = normalize_text(text)
    clean_keyword = normalize_text(keyword)
    if not clean_keyword:
        return False
    if clean_keyword in {"erro", "error", "bug"}:
        negated_pattern = rf"(?<![a-z0-9])(nao|not)\s+(e\s+|is\s+|parece\s+)?{re.escape(clean_keyword)}(?![a-z0-9])"
        if re.search(negated_pattern, clean_text):
            return False
    pattern = rf"(?<![a-z0-9]){re.escape(clean_keyword)}(?![a-z0-9])"
    return re.search(pattern, clean_text) is not None


def explain_ticket(ticket: dict[str, Any], max_items: int = 7) -> list[str]:
    """Return deterministic, keyword-based explanation bullets."""
    text = build_explanation_text(ticket)
    explanations: list[str] = []
    for rule_name, keywords in KEYWORD_RULES.items():
        matches = [kw for kw in keywords if contains_keyword(text, kw)]
        if matches:
            unique_matches = {}
            for match in matches:
                unique_matches.setdefault(normalize_text(match), match)
            sample = ", ".join(sorted(unique_matches.values())[:3])
            explanations.append(f"{rule_name}: matched {sample}")
        if len(explanations) >= max_items:
            break
    if not explanations:
        explanations.append("No strong keyword rule fired; prediction is mostly based on text similarity learned from synthetic examples.")
    return explanations


def load_classifier(model_path: Path | str = MODEL_PATH) -> dict[str, Any]:
    """Load the saved classifier artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python src/generate_data.py` and `python src/train_model.py` first."
        )
    return joblib.load(path)


def _top_scores(pipeline: Any, text: str, limit: int = 4) -> dict[str, float]:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        return {label: round(float(score), 4) for label, score in ranked[:limit]}

    prediction = str(pipeline.predict([text])[0])
    return {prediction: 1.0}


def predict_ticket(ticket: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict all ticket labels and return confidences plus explanations."""
    artifact = artifact or load_classifier()
    text = build_ticket_text(ticket)
    predictions: dict[str, Any] = {}

    for output_name, pipeline in artifact["models"].items():
        label = str(pipeline.predict([text])[0])
        scores = _top_scores(pipeline, text)
        predictions[output_name] = {
            "label": label,
            "confidence": scores.get(label, max(scores.values()) if scores else 1.0),
            "top_scores": scores,
        }

    predictions["explanation"] = explain_ticket(ticket)
    return predictions


def flatten_prediction(prediction: dict[str, Any]) -> dict[str, str]:
    """Return only the predicted class names for simple comparisons/tests."""
    return {
        key: value["label"]
        for key, value in prediction.items()
        if isinstance(value, dict) and "label" in value
    }
