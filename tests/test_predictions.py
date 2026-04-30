from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_data import generate_dataset, write_generation_notes  # noqa: E402
from train_model import train_models  # noqa: E402
from utils import DATA_PATH, MODEL_PATH, build_ticket_text, flatten_prediction, load_classifier, predict_ticket  # noqa: E402


EXPECTED_EXAMPLE_KEYS = {
    "expectedTicketType",
    "expectedPriority",
    "expectedTargetTeam",
    "expectedModule",
    "expectedComponent",
}


def ensure_artifacts() -> dict:
    if not DATA_PATH.exists():
        generate_dataset()
        write_generation_notes(DATA_PATH.parent / "data_generation_notes.md")
    if not MODEL_PATH.exists():
        train_models()
    return load_classifier()


def load_streamlit_examples() -> list[dict]:
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXAMPLES":
                    return ast.literal_eval(node.value)
    raise AssertionError("EXAMPLES was not found in app/streamlit_app.py")


def test_preset_examples_have_expected_evaluation_labels() -> None:
    examples = load_streamlit_examples()

    assert len(examples) >= 15
    for example in examples:
        assert example["label"]
        assert EXPECTED_EXAMPLE_KEYS.issubset(example["expected"])


def test_prediction_is_deterministic() -> None:
    artifact = ensure_artifacts()
    ticket = {
        "ticket_subject": "URGENTE - erro 500 ao emitir fatura",
        "ticket_description": "Estamos sem faturar, aparece exception no IVA e os utilizadores ficam bloqueados.",
        "client_sector": "printing",
        "urgency_signals": "sem faturar; ASAP",
        "previous_classification": "client_misunderstanding / medium / support_level_1",
        "preferred_resolver": "Ana Martins",
    }
    first = flatten_prediction(predict_ticket(ticket, artifact))
    second = flatten_prediction(predict_ticket(ticket, artifact))
    assert first == second


def test_prediction_schema_contains_expected_outputs() -> None:
    artifact = ensure_artifacts()
    prediction = predict_ticket(
        {
            "ticket_subject": "Duvida sobre CRM",
            "ticket_description": "Nao encontro onde mudar o contacto do cliente.",
            "client_sector": "packaging",
            "urgency_signals": "sem urgencia",
            "previous_classification": "",
            "preferred_resolver": "",
        },
        artifact,
    )
    for key in ["ticket_type", "priority", "target_team", "module", "component"]:
        assert key in prediction
        assert "label" in prediction[key]
        assert "confidence" in prediction[key]
        assert "top_scores" in prediction[key]
    assert prediction["explanation"]


def test_description_only_ticket_skips_empty_and_test_metadata() -> None:
    ticket = {
        "label": "Email only - vague CRM problem",
        "ticket_subject": "",
        "ticket_description": "Bom dia, o CRM nao esta a funcionar. Podem ver?",
        "client_sector": "",
        "urgency_signals": "",
        "previous_classification": "",
        "preferred_resolver": "",
        "expected": {
            "expectedTicketType": "needs_triage",
            "expectedTargetTeam": "support_level_1",
        },
    }

    text = build_ticket_text(ticket)

    assert text == "description: Bom dia, o CRM nao esta a funcionar. Podem ver?"
    assert "Email only" not in text
    assert "needs_triage" not in text


def test_urgent_blocking_ticket_gets_high_or_critical_priority() -> None:
    artifact = ensure_artifacts()
    prediction = predict_ticket(
        {
            "ticket_subject": "URGENTE - server timeout e producao parada",
            "ticket_description": "Todos os utilizadores estao bloqueados, VPN/server nao responde e o ambiente esta em baixo. ASAP.",
            "client_sector": "labels",
            "urgency_signals": "producao parada; todos os utilizadores bloqueados",
            "previous_classification": "software_bug / medium / development",
            "preferred_resolver": "",
        },
        artifact,
    )
    assert prediction["priority"]["label"] in {"high", "critical"}


def test_invoice_or_accounting_ticket_maps_to_plausible_module() -> None:
    artifact = ensure_artifacts()
    prediction = predict_ticket(
        {
            "ticket_subject": "Valores de IVA errados na fatura e SAF-T",
            "ticket_description": "A fatura mostra IVA incorreto e o export SAF-T da contabilidade nao bate com o relatorio.",
            "client_sector": "printing",
            "urgency_signals": "fecho do mes",
            "previous_classification": "data_issue / high / technical_support",
            "preferred_resolver": "Helena Rocha",
        },
        artifact,
    )
    assert prediction["module"]["label"] in {"invoicing", "accounting", "reporting"}
