"""Streamlit interface for the ticket classifier."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sample_tickets import SAMPLE_TICKETS, sample_to_ticket  # noqa: E402
from ticket_classifier import MODEL_PATH, PREDICTION_LABELS, flatten_prediction, load_model, predict_ticket  # noqa: E402


@st.cache_resource
def cached_artifact() -> dict:
    return load_model(MODEL_PATH)


def render_confidence(label: str, confidence: float) -> None:
    st.caption(label)
    st.progress(min(max(float(confidence), 0.0), 1.0), text=f"{confidence:.1%}")


def render_ticket_form(sample: dict[str, str]) -> dict[str, str]:
    left, right = st.columns([2, 1])

    with left:
        subject = st.text_input("Ticket subject", value=sample["ticket_subject"])
        description = st.text_area(
            "Ticket description / email body",
            value=sample["ticket_description"],
            height=180,
        )
        previous = st.text_input(
            "Previous/manual classification",
            value=sample["previous_classification"],
        )

    with right:
        client_sector = st.text_input("Client sector", value=sample["client_sector"])
        urgency = st.text_area(
            "Urgency/context notes",
            value=sample["urgency_signals"],
            height=96,
        )
        preferred_resolver = st.text_input(
            "Preferred resolver",
            value=sample["preferred_resolver"],
        )

    return {
        "ticket_subject": subject,
        "ticket_description": description,
        "client_sector": client_sector,
        "urgency_signals": urgency,
        "previous_classification": previous,
        "preferred_resolver": preferred_resolver,
    }


def render_predictions(prediction: dict, show_debug: bool = False, expected: dict | None = None) -> None:
    labels = flatten_prediction(prediction)

    st.divider()
    st.subheader("Predicted Classification")
    columns = st.columns(len(PREDICTION_LABELS))
    for column, output_name in zip(columns, PREDICTION_LABELS):
        with column:
            st.metric(output_name.replace("_", " ").title(), labels[output_name])
            render_confidence("Confidence", prediction[output_name]["confidence"])

    st.subheader("Deterministic Evidence")
    for item in prediction["explanation"]:
        st.write(f"- {item}")

    if show_debug:
        st.subheader("Debug")
        if expected:
            rows = [
                {
                    "field": field,
                    "expected": expected_label,
                    "predicted": labels.get(field, ""),
                    "match": labels.get(field) == expected_label,
                }
                for field, expected_label in expected.items()
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        for output_name in PREDICTION_LABELS:
            st.write(f"**{output_name.replace('_', ' ').title()} top scores**")
            st.json(prediction[output_name]["top_scores"])


def main() -> None:
    st.set_page_config(page_title="Sistrade Ticket Classifier", layout="wide")
    st.title("Sistrade Ticket Classifier")

    with st.sidebar:
        show_debug = st.checkbox("Show debug info", value=False)
        st.caption(f"Model: `{MODEL_PATH}`")

    try:
        artifact = cached_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    blank = {
        "name": "Custom empty ticket",
        "ticket_subject": "",
        "ticket_description": "",
        "client_sector": "",
        "urgency_signals": "",
        "previous_classification": "",
        "preferred_resolver": "",
    }
    samples = [blank, *SAMPLE_TICKETS]
    samples_by_name = {sample["name"]: sample for sample in samples}
    selected_name = st.selectbox(
        "Sample ticket",
        list(samples_by_name),
    )
    selected_sample = samples_by_name[selected_name]

    ticket = render_ticket_form(sample_to_ticket(selected_sample))
    prediction = predict_ticket(ticket, artifact)
    render_predictions(prediction, show_debug=show_debug, expected=selected_sample.get("expected"))


if __name__ == "__main__":
    main()
