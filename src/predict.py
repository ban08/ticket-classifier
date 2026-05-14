"""Command-line predictions for the Sistrade ticket classifier."""

from __future__ import annotations

import argparse
import json

from examples import EXAMPLES, example_to_ticket
from ticket_classifier import flatten_prediction, load_model, predict_ticket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Sistrade support ticket classifications.")
    parser.add_argument("--subject", help="Ticket subject")
    parser.add_argument("--description", help="Ticket description")
    parser.add_argument("--client-sector", default="", help="Client sector, if known")
    parser.add_argument("--urgency", default="", help="Urgency/context notes, if known")
    parser.add_argument("--previous-classification", default="", help="Previous manual classification, if any")
    parser.add_argument("--preferred-resolver", default="", help="Preferred resolver mentioned by the client, if any")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    parser.add_argument("--evaluate-examples", action="store_true", help="Run the built-in debug examples")
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


def evaluate_examples(artifact: dict) -> None:
    matches = 0
    checks = 0
    for example in EXAMPLES:
        ticket = example_to_ticket(example)
        prediction = predict_ticket(ticket, artifact)
        labels = flatten_prediction(prediction)
        print_prediction(ticket, prediction)
        expected = example.get("expected", {})
        if expected:
            print("expected checks:")
            for field, expected_label in expected.items():
                ok = labels.get(field) == expected_label
                matches += int(ok)
                checks += 1
                result = "OK" if ok else "MISS"
                print(f"- {field}: expected={expected_label} predicted={labels.get(field)} {result}")
    if checks:
        print("=" * 78)
        print(f"Debug example checks: {matches}/{checks} matched")


def main() -> None:
    args = parse_args()
    artifact = load_model()
    if args.evaluate_examples:
        evaluate_examples(artifact)
        return

    ticket = ticket_from_args(args)
    tickets = [ticket] if ticket else [example_to_ticket(example) for example in EXAMPLES[:3]]

    for item in tickets:
        prediction = predict_ticket(item, artifact)
        print_prediction(item, prediction, args.json)


if __name__ == "__main__":
    main()
