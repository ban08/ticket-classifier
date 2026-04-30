"""Command-line predictions for the Sistrade ticket classifier."""

from __future__ import annotations

import argparse
import json

from utils import flatten_prediction, load_classifier, predict_ticket


SAMPLE_TICKETS = [
    {
        "ticket_subject": "URGENTE - erro 500 ao emitir fatura",
        "ticket_description": "Bom dia, estamos sem faturar desde esta manha. O invoice fica bloqueado e aparece exception no IVA. ASAP.",
        "client_sector": "printing",
        "urgency_signals": "sem faturar; producao parada",
        "previous_classification": "client_misunderstanding / medium / support_level_1",
        "preferred_resolver": "Ana Martins",
    },
    {
        "ticket_subject": "Duvida sobre permissao no CRM",
        "ticket_description": "O utilizador nao encontra onde alterar o perfil do cliente. Talvez seja so duvida de menu.",
        "client_sector": "packaging",
        "urgency_signals": "sem urgencia",
        "previous_classification": "software_bug / high / development",
        "preferred_resolver": "",
    },
    {
        "ticket_subject": "Feature request para dashboard de producao",
        "ticket_description": "Gostavamos de uma nova opcao no dashboard para ver KPI por ordem de fabrico. Nao e erro, pode ser analisado.",
        "client_sector": "labels",
        "urgency_signals": "melhoria para analisar",
        "previous_classification": "configuration_request / low / technical_support",
        "preferred_resolver": "Marta Silva",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Sistrade support ticket classifications.")
    parser.add_argument("--subject", help="Ticket subject")
    parser.add_argument("--description", help="Ticket description")
    parser.add_argument("--client-sector", default="", help="Client sector, if known")
    parser.add_argument("--urgency", default="", help="Urgency/context notes, if known")
    parser.add_argument("--previous-classification", default="", help="Previous manual classification, if any")
    parser.add_argument("--preferred-resolver", default="", help="Preferred resolver mentioned by the client, if any")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    return parser.parse_args()


def ticket_from_args(args: argparse.Namespace) -> dict | None:
    if not args.subject and not args.description:
        return None
    return {
        "ticket_subject": args.subject or "",
        "ticket_description": args.description or "",
        "client_sector": args.client_sector,
        "urgency_signals": args.urgency,
        "previous_classification": args.previous_classification,
        "preferred_resolver": args.preferred_resolver,
    }


def print_prediction(ticket: dict, prediction: dict, full_json: bool = False) -> None:
    print("=" * 78)
    print(ticket["ticket_subject"])
    if full_json:
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
        return
    labels = flatten_prediction(prediction)
    for key, value in labels.items():
        confidence = prediction[key]["confidence"]
        print(f"{key:12s}: {value:24s} confidence={confidence:.3f}")
    print("explanation:")
    for item in prediction["explanation"]:
        print(f"- {item}")


def main() -> None:
    args = parse_args()
    artifact = load_classifier()
    ticket = ticket_from_args(args)
    tickets = [ticket] if ticket else SAMPLE_TICKETS

    for item in tickets:
        prediction = predict_ticket(item, artifact)
        print_prediction(item, prediction, args.json)


if __name__ == "__main__":
    main()
