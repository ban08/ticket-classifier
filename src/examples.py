"""Realistic tickets used by the CLI and Streamlit demo."""

from __future__ import annotations

from typing import Any

from ticket_classifier import TICKET_FIELDS


EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "Urgent invoicing bug",
        "ticket_subject": "URGENTE - erro 500 ao emitir fatura",
        "ticket_description": "Estamos sem faturar desde esta manha. O invoice fica bloqueado e aparece exception no IVA.",
        "client_sector": "printing",
        "urgency_signals": "sem faturar; ASAP; deadline hoje",
        "previous_classification": "client_misunderstanding / medium / support_level_1",
        "preferred_resolver": "Ana Martins",
        "expected": {"ticket_type": "software_bug", "priority": "critical", "module": "invoicing"},
    },
    {
        "name": "CRM permission setup",
        "ticket_subject": "Configurar perfis de aprovacao no CRM",
        "ticket_description": "Precisamos ajustar perfis, role e aprovacoes para que os supervisores possam editar clientes no CRM sem acesso total ao resto do modulo.",
        "client_sector": "packaging",
        "urgency_signals": "afeta uma equipa; precisamos esta semana",
        "previous_classification": "client_misunderstanding / medium / support_level_1",
        "preferred_resolver": "Rui Costa",
        "expected": {"ticket_type": "configuration_request", "priority": "medium", "module": "user_permissions"},
    },
    {
        "name": "Stock import duplicates",
        "ticket_subject": "Importacao CSV duplicou registos",
        "ticket_description": "Depois do import de artigos via CSV ficamos com produtos duplicados e saldos errados em stock.",
        "client_sector": "packaging",
        "urgency_signals": "cliente a espera; afeta uma equipa",
        "previous_classification": "",
        "preferred_resolver": "",
        "expected": {"ticket_type": "data_issue", "priority": "high", "module": "stock"},
    },
    {
        "name": "Server VPN timeout",
        "ticket_subject": "Timeout no acesso ao Sistrade",
        "ticket_description": "Varios utilizadores estao bloqueados por VPN/server timeout. O ambiente fica lento e cai.",
        "client_sector": "labels",
        "urgency_signals": "varios utilizadores bloqueados; urgente",
        "previous_classification": "",
        "preferred_resolver": "",
        "expected": {"ticket_type": "infrastructure_problem", "priority": "critical", "module": "general"},
    },
    {
        "name": "Training request",
        "ticket_subject": "Ajuda para novos utilizadores",
        "ticket_description": "Temos 3 pessoas novas na equipa comercial e queriamos uma sessao de formacao sobre contactos e pipeline.",
        "client_sector": "printing",
        "urgency_signals": "quando possivel",
        "previous_classification": "",
        "preferred_resolver": "",
        "expected": {"ticket_type": "training_needed", "priority": "low", "module": "CRM"},
    },
    {
        "name": "Dashboard improvement",
        "ticket_subject": "Nova opcao no dashboard",
        "ticket_description": "Could you add a new dashboard option to compare KPIs by production line? No urgency, this is an improvement request.",
        "client_sector": "industrial manufacturing",
        "urgency_signals": "melhoria para analisar",
        "previous_classification": "",
        "preferred_resolver": "",
        "expected": {"ticket_type": "feature_request", "priority": "low", "module": "reporting"},
    },
]


def example_to_ticket(example: dict[str, Any]) -> dict[str, str]:
    return {field: str(example.get(field, "")) for field in TICKET_FIELDS}
