"""Streamlit web app for the Sistrade ticket classification POC."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import MODEL_PATH, flatten_prediction, load_classifier, predict_ticket  # noqa: E402


# Preset labels are only demo scenario names shown in the dropdown. They are not
# sent to the classifier. The model input is built only from the user-provided
# ticket fields below. The nested expected classification is development/test
# metadata and must not be included in the prediction payload.
EXAMPLES = [
    {
        "label": "Structured - urgent invoicing bug",
        "ticketSubject": "URGENTE - erro 500 ao emitir fatura",
        "ticketDescription": "Estamos sem faturar desde esta manha. O invoice fica bloqueado e aparece exception no IVA. ASAP.",
        "clientSector": "printing",
        "urgencyContextNotes": "sem faturar; deadline hoje",
        "preferredResolver": "Ana Martins",
        "previousManualClassification": "client_misunderstanding / medium / support_level_1",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "development",
            "expectedPriority": "critical",
            "expectedModule": "invoicing",
            "expectedComponent": "invoice_emission",
            "reasoningNote": "Invoicing is blocked by an error, so this should be escalated quickly.",
        },
    },
    {
        "label": "Structured - CRM permission/configuration",
        "ticketSubject": "Bug no CRM - utilizador nao consegue alterar cliente",
        "ticketDescription": "O utilizador diz que e bug, mas parece que nao tem permissao para editar o perfil do cliente no CRM.",
        "clientSector": "packaging",
        "urgencyContextNotes": "sem urgencia; novo utilizador",
        "preferredResolver": "Rui Costa",
        "previousManualClassification": "software_bug / high / development",
        "expected": {
            "expectedTicketType": "configuration_request",
            "expectedTargetTeam": "support_level_1",
            "expectedPriority": "low",
            "expectedModule": "CRM",
            "expectedComponent": "access_rights",
            "reasoningNote": "Client calls it a bug, but the evidence points to access/profile setup.",
        },
    },
    {
        "label": "Structured - approval workflow setup",
        "ticketSubject": "Configurar workflow de aprovacao",
        "ticketDescription": "Precisamos configurar novas regras de aprovacao para perfis de compras e producao.",
        "clientSector": "industrial manufacturing",
        "urgencyContextNotes": "precisamos esta semana",
        "preferredResolver": "Marta Silva",
        "previousManualClassification": "support_level_1 / low / client_misunderstanding",
        "expected": {
            "expectedTicketType": "configuration_request",
            "expectedTargetTeam": "technical_support",
            "expectedPriority": "medium",
            "expectedModule": "user_permissions",
            "expectedComponent": "approvals",
            "reasoningNote": "A planned setup request with timing pressure, not a production defect.",
        },
    },
    {
        "label": "Structured - integration API failure",
        "ticketSubject": "API ecommerce retorna 401 nas encomendas",
        "ticketDescription": "Desde a troca de certificado, a integracao ecommerce deixou de importar encomendas. O webservice responde 401.",
        "clientSector": "labels",
        "urgencyContextNotes": "loja online parada desde ontem",
        "preferredResolver": "Helena Rocha",
        "previousManualClassification": "data_issue / medium / technical_support",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "technical_support",
            "expectedPriority": "high",
            "expectedModule": "integrations",
            "expectedComponent": "api",
            "reasoningNote": "Authentication failure in an integration that blocks order import.",
        },
    },
    {
        "label": "Description only - vague CRM problem",
        "ticketSubject": "",
        "ticketDescription": "Bom dia, o CRM nao esta a funcionar. Podem ver?",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "needs_triage",
            "expectedTargetTeam": "support_level_1",
            "expectedPriority": "medium",
            "expectedModule": "CRM",
            "expectedComponent": "unknown",
            "reasoningNote": "Very little information. Support should triage before escalation.",
        },
    },
    {
        "label": "Description only - urgent production down",
        "ticketSubject": "",
        "ticketDescription": "URGENTE. A producao esta parada e todos os utilizadores estao bloqueados. Nao conseguimos abrir ordens de fabrico.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "development",
            "expectedPriority": "critical",
            "expectedModule": "production",
            "expectedComponent": "work_orders",
            "reasoningNote": "Only email body exists, but it clearly states production is blocked.",
        },
    },
    {
        "label": "Description only - too little information",
        "ticketSubject": "",
        "ticketDescription": "Nao da. Podem ajudar?",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "needs_triage",
            "expectedTargetTeam": "support_level_1",
            "expectedPriority": "low",
            "expectedModule": "unknown",
            "expectedComponent": "unknown",
            "reasoningNote": "Insufficient information; should request clarification.",
        },
    },
    {
        "label": "Description only - English feature request",
        "ticketSubject": "",
        "ticketDescription": "Could you add a new dashboard option to compare KPIs by production line? No urgency, this is an improvement request.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "feature_request",
            "expectedTargetTeam": "account_management",
            "expectedPriority": "low",
            "expectedModule": "reporting",
            "expectedComponent": "dashboards",
            "reasoningNote": "Explicit non-urgent improvement request.",
        },
    },
    {
        "label": "Subject + description - data import duplicates",
        "ticketSubject": "Importacao CSV duplicou registos",
        "ticketDescription": "Depois do import de artigos via CSV ficamos com produtos duplicados e saldos errados em stock.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "data_issue",
            "expectedTargetTeam": "technical_support",
            "expectedPriority": "medium",
            "expectedModule": "stock",
            "expectedComponent": "stock_movements",
            "reasoningNote": "Import created duplicate records and stock inconsistencies.",
        },
    },
    {
        "label": "Subject + description - slow reporting",
        "ticketSubject": "Reports are very slow",
        "ticketDescription": "The sales dashboard takes more than 2 minutes to load since Monday. Users can work, but reporting is painful.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "technical_support",
            "expectedPriority": "medium",
            "expectedModule": "reporting",
            "expectedComponent": "performance",
            "reasoningNote": "Performance degradation affects reporting but is not fully blocking operations.",
        },
    },
    {
        "label": "Subject + description - duplicate follow-up",
        "ticketSubject": "Follow-up ticket 4281 - invoice still blocked",
        "ticketDescription": "This is the same issue as yesterday. The invoice validation still fails for IVA on export.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "support_level_1",
            "expectedPriority": "medium",
            "expectedModule": "invoicing",
            "expectedComponent": "tax_validation",
            "reasoningNote": "Follow-up should be linked to the existing ticket before new routing.",
        },
    },
    {
        "label": "Subject + description - server/VPN timeout",
        "ticketSubject": "Timeout no acesso ao Sistrade",
        "ticketDescription": "Varios utilizadores estao bloqueados por VPN/server timeout. O ambiente fica lento e cai.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "infrastructure_problem",
            "expectedTargetTeam": "infrastructure",
            "expectedPriority": "high",
            "expectedModule": "general",
            "expectedComponent": "performance",
            "reasoningNote": "Multiple users blocked by VPN/server instability.",
        },
    },
    {
        "label": "Email messy - mixed language bug/config",
        "ticketSubject": "BUG? user cannot approve PO",
        "ticketDescription": "Hello, o Antonio cannot approve purchase orders. It says sem permissao. We changed his role last week, maybe profile missing?",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "configuration_request",
            "expectedTargetTeam": "support_level_1",
            "expectedPriority": "medium",
            "expectedModule": "user_permissions",
            "expectedComponent": "roles_profiles",
            "reasoningNote": "Mixed-language email calls it a bug, but permission and role clues suggest configuration.",
        },
    },
    {
        "label": "Email messy - long Portuguese invoicing",
        "ticketSubject": "Problema nas faturas do cliente XPTO",
        "ticketDescription": "Bom dia,\n\nDesde a atualizacao de ontem, quando tentamos emitir fatura para encomendas com portes, o valor do IVA fica diferente do relatorio. Ja tentamos repetir em dois postos. Nao aparece erro 500, mas o documento fica com valores incorretos e a contabilidade nao quer fechar o dia assim.\n\nPodem analisar ainda hoje?",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "development",
            "expectedPriority": "high",
            "expectedModule": "invoicing",
            "expectedComponent": "tax_validation",
            "reasoningNote": "Long email gives enough evidence for a high-priority invoicing/tax defect.",
        },
    },
    {
        "label": "Email messy - English API payload",
        "ticketSubject": "EDI orders not arriving",
        "ticketDescription": "Hi team,\n\nOur customer says the EDI integration stopped at 03:20. The API returns 200 but the orders are not created in Sistrade. Attached is a sample payload and response. Could be mapping after the new field customerReference.\n\nThanks.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "software_bug",
            "expectedTargetTeam": "technical_support",
            "expectedPriority": "high",
            "expectedModule": "integrations",
            "expectedComponent": "edi",
            "reasoningNote": "API succeeds but orders are not created, likely integration mapping.",
        },
    },
    {
        "label": "Email messy - non-urgent training",
        "ticketSubject": "Ajuda para novos utilizadores",
        "ticketDescription": "Temos 3 pessoas novas na equipa comercial e queriamos uma pequena sessao de formacao sobre contactos, oportunidades e pipeline. Pode ser para a proxima semana.",
        "clientSector": "",
        "urgencyContextNotes": "",
        "preferredResolver": "",
        "previousManualClassification": "",
        "expected": {
            "expectedTicketType": "training_needed",
            "expectedTargetTeam": "training_consulting",
            "expectedPriority": "low",
            "expectedModule": "CRM",
            "expectedComponent": "commercial_pipeline",
            "reasoningNote": "Non-urgent training request for new CRM users.",
        },
    },
]


@st.cache_resource
def cached_artifact() -> dict:
    return load_classifier(MODEL_PATH)


def confidence_bar(label: str, confidence: float) -> None:
    st.caption(label)
    st.progress(min(max(float(confidence), 0.0), 1.0), text=f"{confidence:.1%}")


EVALUATION_FIELDS = [
    ("ticket_type", "expectedTicketType", "Ticket Type"),
    ("priority", "expectedPriority", "Priority"),
    ("target_team", "expectedTargetTeam", "Target Team"),
    ("module", "expectedModule", "Module"),
    ("component", "expectedComponent", "Component"),
]


def normalize_label(value: Any) -> str:
    return str(value or "").strip().casefold()


def example_to_ticket(example: dict[str, Any]) -> dict[str, str]:
    """Return only realistic user/client fields for classifier input.

    The preset dropdown label, reasoning note, and expected classification are
    intentionally excluded so evaluation cannot leak ground truth to the model.
    """
    return {
        "ticket_subject": example["ticketSubject"],
        "ticket_description": example["ticketDescription"],
        "client_sector": example["clientSector"],
        "urgency_signals": example["urgencyContextNotes"],
        "previous_classification": example["previousManualClassification"],
        "preferred_resolver": example["preferredResolver"],
    }


def compare_prediction(example_name: str, expected: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    predicted = flatten_prediction(prediction)
    field_results = []
    for prediction_key, expected_key, display_name in EVALUATION_FIELDS:
        expected_value = expected[expected_key]
        predicted_value = predicted[prediction_key]
        matches = normalize_label(expected_value) == normalize_label(predicted_value)
        field_results.append(
            {
                "field": display_name,
                "expected": expected_value,
                "predicted": predicted_value,
                "confidence": prediction[prediction_key]["confidence"],
                "match": matches,
            }
        )

    all_match = all(item["match"] for item in field_results)
    type_match = field_results[0]["match"]
    team_match = field_results[2]["match"]
    if all_match:
        result = "PASS"
    elif type_match or team_match:
        result = "PARTIAL"
    else:
        result = "FAIL"

    return {
        "example": example_name,
        "result": result,
        "fields": field_results,
        "reasoningNote": expected.get("reasoningNote", ""),
    }


def evaluate_example(example_name: str, ticket: dict[str, Any], expected: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    prediction = predict_ticket(ticket, artifact)
    return compare_prediction(example_name, expected, prediction)


def evaluate_all_examples(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        evaluate_example(example["label"], example_to_ticket(example), example["expected"], artifact)
        for example in EXAMPLES
    ]


def evaluation_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        by_field = {item["field"]: item for item in result["fields"]}
        rows.append(
            {
                "example name": result["example"],
                "expected type": by_field["Ticket Type"]["expected"],
                "predicted type": by_field["Ticket Type"]["predicted"],
                "expected priority": by_field["Priority"]["expected"],
                "predicted priority": by_field["Priority"]["predicted"],
                "expected team": by_field["Target Team"]["expected"],
                "predicted team": by_field["Target Team"]["predicted"],
                "expected module": by_field["Module"]["expected"],
                "predicted module": by_field["Module"]["predicted"],
                "expected component": by_field["Component"]["expected"],
                "predicted component": by_field["Component"]["predicted"],
                "result": result["result"],
            }
        )
    return rows


def render_single_evaluation(result: dict[str, Any]) -> None:
    st.metric("Overall result", result["result"])
    rows = [
        {
            "field": item["field"],
            "expected": item["expected"],
            "predicted": item["predicted"],
            "confidence": f"{item['confidence']:.1%}",
            "match": "MATCH" if item["match"] else "MISMATCH",
        }
        for item in result["fields"]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if result["reasoningNote"]:
        st.caption(f"Expected-label note: {result['reasoningNote']}")


def render_diagnostic_guidance(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    total = len(results)
    fail_counts = {field_name: 0 for _, _, field_name in EVALUATION_FIELDS}
    vague_failures = 0
    for result in results:
        for item in result["fields"]:
            if not item["match"]:
                fail_counts[item["field"]] += 1
        if result["result"] != "PASS":
            expected_type = result["fields"][0]["expected"]
            if normalize_label(expected_type) == "needs_triage" or "vague" in normalize_label(result["example"]):
                vague_failures += 1

    guidance = []
    if fail_counts["Ticket Type"] / total >= 0.3:
        guidance.append("Many examples miss ticket type: review label definitions, add clearer examples, or add few-shot examples to the prompt/workflow.")
    if fail_counts["Priority"] / total >= 0.3:
        guidance.append("Many examples miss priority: standardize priority rules before changing the model.")
    if (fail_counts["Module"] + fail_counts["Component"]) / (total * 2) >= 0.3:
        guidance.append("Many examples miss module/component: add ERP module/component descriptions, manuals, or RAG context.")
    non_pass = [result for result in results if result["result"] != "PASS"]
    if non_pass and vague_failures == len(non_pass):
        guidance.append("Only vague tickets are failing: this is acceptable uncertainty; route them as needs_triage instead of forcing precision.")
    if not guidance:
        guidance.append("No broad failure pattern stands out. Inspect individual mismatches before changing labels or model behavior.")

    for item in guidance:
        st.write(f"- {item}")


def render_debug_panel(example: dict[str, Any], current_ticket: dict[str, Any], artifact: dict[str, Any]) -> None:
    st.subheader("Debug evaluation")
    st.caption(
        "Expected labels are local evaluation metadata only. They are never sent to the classifier. "
        "A mismatch is a signal to inspect the case, not automatic proof that the model is bad."
    )

    st.info(
        "What to do with failures: do not blindly train the AI just because one example failed. "
        "First check whether the expected label is fair. Then check whether the input contains enough information. "
        "Then improve the system prompt, label definitions, or examples. Only consider training/fine-tuning if "
        "failures are frequent, consistent, and happen on well-defined cases. Some tickets are ambiguous and will fail; that is normal."
    )

    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Evaluate example"):
            st.session_state["current_evaluation"] = evaluate_example(
                example["label"],
                current_ticket,
                example["expected"],
                artifact,
            )
    with button_cols[1]:
        if st.button("Evaluate all examples"):
            st.session_state["all_evaluations"] = evaluate_all_examples(artifact)

    if "current_evaluation" in st.session_state:
        st.write("**Current example result**")
        render_single_evaluation(st.session_state["current_evaluation"])

    if "all_evaluations" in st.session_state:
        st.write("**All examples summary**")
        results = st.session_state["all_evaluations"]
        st.dataframe(evaluation_summary_rows(results), hide_index=True, use_container_width=True)
        counts = {label: sum(1 for result in results if result["result"] == label) for label in ["PASS", "PARTIAL", "FAIL"]}
        st.caption(f"Summary: {counts['PASS']} PASS, {counts['PARTIAL']} PARTIAL, {counts['FAIL']} FAIL")
        st.write("**Diagnostic guidance**")
        render_diagnostic_guidance(results)


def main() -> None:
    st.set_page_config(page_title="Sistrade Ticket Classifier", layout="wide")
    st.title("Sistrade Ticket Classifier POC")

    with st.sidebar:
        st.subheader("POC notes")
        st.write("Classic local ML model trained on synthetic support tickets.")
        st.write("It assists triage; it does not answer clients automatically.")
        st.write("Preset examples fill the form only; the example name is not used as model evidence.")
        st.write(f"Model path: `{MODEL_PATH}`")
        show_debug = st.checkbox("Debug/evaluation mode", value=False)

    try:
        artifact = cached_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    labels_by_example = {example["label"]: example for example in EXAMPLES}
    selected = st.selectbox(
        "Example",
        list(labels_by_example),
        help="Named preset scenarios for demos and tests. The selected name is not sent to the classifier.",
    )
    example = labels_by_example[selected]

    left, right = st.columns([2, 1])
    with left:
        subject = st.text_input("Ticket subject", value=example["ticketSubject"])
        st.caption("For email-style tickets, paste the email body here. Other fields are optional.")
        description = st.text_area("Ticket description / email body", value=example["ticketDescription"], height=180)
        previous = st.text_input("Previous/manual classification", value=example["previousManualClassification"])
    with right:
        st.caption("Empty fields are valid. Use them only when this context is known.")
        client_sector = st.text_input("Client sector", value=example["clientSector"])
        urgency = st.text_area("Urgency/context notes", value=example["urgencyContextNotes"], height=96)
        preferred_resolver = st.text_input("Preferred resolver", value=example["preferredResolver"])

    # This payload intentionally uses only real ticket fields a user/client could
    # provide. It excludes the preset example label and test-only expected data.
    ticket = {
        "ticket_subject": subject,
        "ticket_description": description,
        "client_sector": client_sector,
        "urgency_signals": urgency,
        "previous_classification": previous,
        "preferred_resolver": preferred_resolver,
    }

    prediction = predict_ticket(ticket, artifact)
    labels = flatten_prediction(prediction)

    st.divider()
    st.subheader("Predicted classification")
    cols = st.columns(5)
    for col, output_name in zip(cols, ["ticket_type", "priority", "target_team", "module", "component"]):
        with col:
            st.metric(output_name.replace("_", " ").title(), labels[output_name])
            confidence_bar("Confidence", prediction[output_name]["confidence"])

    st.subheader("Top confidence scores")
    score_cols = st.columns(3)
    for idx, output_name in enumerate(["ticket_type", "priority", "target_team", "module", "component"]):
        with score_cols[idx % 3]:
            st.write(f"**{output_name.replace('_', ' ').title()}**")
            for label, score in prediction[output_name]["top_scores"].items():
                st.progress(score, text=f"{label}: {score:.1%}")

    st.subheader("Deterministic explanation")
    for item in prediction["explanation"]:
        st.write(f"- {item}")

    if show_debug:
        render_debug_panel(example, ticket, artifact)


if __name__ == "__main__":
    main()
