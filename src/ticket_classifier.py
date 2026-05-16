"""Core helpers for the Sistrade ticket classification project."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import unicodedata

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ai_generated_sistrade_tickets.csv"
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

OPERATIONAL_TYPES = {"software_bug", "data_issue", "infrastructure_problem"}

URGENT_SIGNAL_KEYWORDS = [
    "urgente",
    "urgent",
    "asap",
    "sem faturar",
    "cannot invoice",
    "nao conseguimos faturar",
    "produção parada",
    "producao parada",
    "parou a producao",
    "fecho do mes",
    "month end",
    "deadline hoje",
    "deadline today",
    "cliente parado",
    "todos bloqueados",
    "all users blocked",
    "varios utilizadores",
    "muitos utilizadores",
    "falha geral",
]

INCIDENT_KEYWORDS = [
    "erro",
    "error",
    "exception",
    "stacktrace",
    "timeout",
    "bloqueado",
    "blocked",
    "nao grava",
    "nao abre",
    "crash",
    "lento",
    "slow",
    "duplicados",
    "duplicou",
    "saldos errados",
    "wrong totals",
    "vpn",
    "servidor",
    "server",
]

LOW_IMPACT_KEYWORDS = [
    "quando possivel",
    "sem urgencia",
    "sem urgência",
    "pode esperar",
    "para roadmap",
    "improvement request",
    "melhoria",
    "feature request",
]

FINANCIAL_KEYWORDS = [
    "fatura",
    "factura",
    "invoice",
    "iva",
    "e-invoice",
    "saf-t",
    "saft",
    "credito",
    "credit note",
    "pagamento",
    "payment",
    "fecho do mes",
]

OPERATIONAL_KEYWORDS = [
    "stock",
    "inventario",
    "inventário",
    "armazem",
    "armazém",
    "shop floor",
    "ordem de fabrico",
    "work order",
    "edi",
    "api",
    "webservice",
    "integração",
    "integracao",
]

CONFIGURATION_KEYWORDS = [
    "configurar",
    "configuracao",
    "configuração",
    "parametrizacao",
    "parametrização",
    "perfil",
    "role",
    "workflow",
    "template",
]

PERMISSION_KEYWORDS = [
    "permissoes",
    "permissões",
    "permissao",
    "permissão",
    "perfil",
    "role",
    "acesso",
    "acessos",
    "approval",
    "approvals",
    "password",
]


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


def has_any_keyword(text: str, keywords: list[str]) -> bool:
    return any(contains_keyword(text, keyword) for keyword in keywords)


def is_uncertain(scores: dict[str, float], margin: float = 0.14) -> bool:
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) < 2:
        return False
    return (ranked[0] - ranked[1]) <= margin


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {label: score / total for label, score in scores.items()}


def apply_multipliers(scores: dict[str, float], multipliers: dict[str, float]) -> dict[str, float]:
    adjusted = {label: score * multipliers.get(label, 1.0) for label, score in scores.items()}
    return normalize_scores(adjusted)


def build_risk_context(ticket: dict[str, Any], raw_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    text = build_ticket_text(ticket)
    return {
        "text": text,
        "urgent_signals": has_any_keyword(text, URGENT_SIGNAL_KEYWORDS),
        "incident_signals": has_any_keyword(text, INCIDENT_KEYWORDS),
        "low_impact_signals": has_any_keyword(text, LOW_IMPACT_KEYWORDS),
        "financial_signals": has_any_keyword(text, FINANCIAL_KEYWORDS),
        "operational_signals": has_any_keyword(text, OPERATIONAL_KEYWORDS),
        "configuration_signals": has_any_keyword(text, CONFIGURATION_KEYWORDS),
        "permission_signals": has_any_keyword(text, PERMISSION_KEYWORDS),
        "raw_priority": raw_predictions["priority"]["label"],
        "raw_ticket_type": raw_predictions["ticket_type"]["label"],
        "raw_module": raw_predictions["module"]["label"],
    }


def adjust_scores_for_business_risk(
    output_name: str,
    scores: dict[str, float],
    context: dict[str, Any],
) -> tuple[str, dict[str, float], bool]:
    adjusted = dict(scores)
    changed = False

    if output_name == "priority":
        multipliers = {"low": 0.97, "medium": 1.0, "high": 1.03, "critical": 1.05}
        if context["urgent_signals"] or context["incident_signals"]:
            if is_uncertain(scores, margin=0.18):
                multipliers.update({"low": 0.82, "medium": 0.94, "high": 1.10, "critical": 1.18})
            else:
                multipliers.update({"low": 0.92, "medium": 0.98, "high": 1.05, "critical": 1.08})
        if context["financial_signals"] and not context["low_impact_signals"]:
            multipliers["high"] = multipliers.get("high", 1.0) * 1.05
            multipliers["critical"] = multipliers.get("critical", 1.0) * 1.08
        adjusted = apply_multipliers(scores, multipliers)

    elif output_name == "ticket_type":
        if context["incident_signals"] or context["urgent_signals"]:
            multipliers = {
                "software_bug": 1.08,
                "data_issue": 1.06,
                "infrastructure_problem": 1.08,
                "client_misunderstanding": 0.92,
                "training_needed": 0.90,
                "feature_request": 0.90,
            }
            if is_uncertain(scores, margin=0.16):
                adjusted = apply_multipliers(scores, multipliers)
        elif context["low_impact_signals"] and not context["incident_signals"]:
            adjusted = apply_multipliers(
                scores,
                {
                    "feature_request": 1.06,
                    "training_needed": 1.05,
                    "client_misunderstanding": 1.04,
                    "software_bug": 0.96,
                    "infrastructure_problem": 0.95,
                },
            )
        if context["configuration_signals"] and not context["incident_signals"] and is_uncertain(scores, margin=0.20):
            adjusted = apply_multipliers(
                adjusted,
                {
                    "configuration_request": 1.14,
                    "client_misunderstanding": 0.94,
                    "feature_request": 0.97,
                },
            )

    elif output_name == "module":
        multipliers: dict[str, float] = {}
        if context["financial_signals"]:
            multipliers.update({"invoicing": 1.10, "accounting": 1.07, "reporting": 0.96})
        if context["operational_signals"]:
            multipliers.update({"stock": 1.07, "production": 1.07, "integrations": 1.08})
        if context["permission_signals"]:
            multipliers.update({"user_permissions": 1.14, "CRM": 0.95, "general": 0.95})
        if multipliers and is_uncertain(scores, margin=0.17):
            adjusted = apply_multipliers(scores, multipliers)

    elif output_name == "target_team":
        risky_case = context["raw_priority"] in {"high", "critical"} or context["raw_ticket_type"] in OPERATIONAL_TYPES
        if risky_case and is_uncertain(scores, margin=0.20):
            multipliers = {
                "support_level_1": 0.90,
                "training_consulting": 0.88,
                "account_management": 0.88,
                "technical_support": 1.06,
                "development": 1.08,
                "infrastructure": 1.08,
            }
            if context["raw_ticket_type"] == "infrastructure_problem":
                multipliers["infrastructure"] = 1.14
            adjusted = apply_multipliers(scores, multipliers)

    raw_label = max(scores, key=scores.get)
    adjusted_label = max(adjusted, key=adjusted.get)
    if adjusted_label != raw_label:
        changed = True
    return adjusted_label, {label: round(float(score), 4) for label, score in adjusted.items()}, changed


def predict_ticket(ticket: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_model()
    text = build_ticket_text(ticket)
    prediction: dict[str, Any] = {}
    raw_predictions: dict[str, dict[str, Any]] = {}

    for output_name in PREDICTION_LABELS:
        pipeline = artifact["models"][output_name]
        label = str(pipeline.predict([text])[0])
        scores = top_scores(pipeline, text)
        raw_predictions[output_name] = {
            "label": label,
            "confidence": scores.get(label, max(scores.values())),
            "top_scores": scores,
        }

    risk_context = build_risk_context(ticket, raw_predictions)

    for output_name in PREDICTION_LABELS:
        raw_item = raw_predictions[output_name]
        label, adjusted_scores, adjusted = adjust_scores_for_business_risk(
            output_name,
            raw_item["top_scores"],
            risk_context,
        )
        prediction[output_name] = {
            "label": label,
            "confidence": adjusted_scores.get(label, max(adjusted_scores.values())),
            "top_scores": adjusted_scores,
            "raw_label": raw_item["label"],
            "risk_adjusted": adjusted,
        }

    prediction["explanation"] = explain_ticket(ticket)
    return prediction


def flatten_prediction(prediction: dict[str, Any]) -> dict[str, str]:
    return {key: prediction[key]["label"] for key in PREDICTION_LABELS}
