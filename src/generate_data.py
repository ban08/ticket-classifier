"""Generate synthetic Sistrade-like support tickets for the ML POC."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

from ticket_classifier import DATA_PATH, EXPECTED_DATA_COLUMNS, RANDOM_STATE


N_TICKETS = 1400

CLIENTS = [
    ("GrafiNorte SA", "printing"),
    ("LusoPack Embalagens", "packaging"),
    ("TextilMinho", "textile"),
    ("EtiquetaFina", "labels"),
    ("FlexoPrime", "printing"),
    ("Cartonagem Douro", "packaging"),
    ("PlastiCentro", "plastics"),
    ("PrintWorks Porto", "printing"),
    ("MetalLabel", "industrial manufacturing"),
    ("AlfaForms", "forms and labels"),
    ("EuroPackaging PT", "packaging"),
    ("Braga Print House", "printing"),
]

RESOLVERS = ["Ana Martins", "Joao Pereira", "Marta Silva", "Rui Costa", "Helena Rocha", "Miguel Santos"]

MODULE_COMPONENTS = {
    "CRM": ["customer_cards", "opportunities", "complaints", "commercial_pipeline"],
    "invoicing": ["invoice_emission", "tax_validation", "e_invoice", "credit_notes"],
    "stock": ["inventory_counts", "warehouse_transfer", "barcode_labels", "stock_movements"],
    "production": ["work_orders", "scheduling", "shop_floor", "cost_calculation"],
    "accounting": ["journal_entries", "vat_report", "bank_reconciliation", "saft_export"],
    "reporting": ["dashboards", "custom_reports", "excel_export", "kpi_analysis"],
    "user_permissions": ["roles_profiles", "access_rights", "approvals", "password_policy"],
    "integrations": ["api", "edi", "webservices", "external_accounting"],
    "general": ["login", "notifications", "navigation", "performance"],
}

TICKET_TYPES = [
    "software_bug",
    "client_misunderstanding",
    "configuration_request",
    "data_issue",
    "training_needed",
    "infrastructure_problem",
    "feature_request",
]

PRIORITIES = ["low", "medium", "high", "critical"]
TEAMS = [
    "support_level_1",
    "technical_support",
    "development",
    "infrastructure",
    "training_consulting",
    "account_management",
]

TYPE_WEIGHTS = {
    "software_bug": 0.22,
    "client_misunderstanding": 0.18,
    "configuration_request": 0.16,
    "data_issue": 0.15,
    "training_needed": 0.11,
    "infrastructure_problem": 0.10,
    "feature_request": 0.08,
}

TYPE_MODULE_WEIGHTS = {
    "software_bug": {
        "invoicing": 0.20,
        "production": 0.20,
        "stock": 0.14,
        "accounting": 0.13,
        "integrations": 0.13,
        "reporting": 0.10,
        "CRM": 0.06,
        "general": 0.04,
    },
    "client_misunderstanding": {
        "general": 0.18,
        "invoicing": 0.16,
        "CRM": 0.15,
        "stock": 0.13,
        "reporting": 0.12,
        "production": 0.10,
        "user_permissions": 0.10,
        "accounting": 0.06,
    },
    "configuration_request": {
        "user_permissions": 0.28,
        "invoicing": 0.17,
        "production": 0.15,
        "integrations": 0.14,
        "reporting": 0.10,
        "CRM": 0.08,
        "stock": 0.08,
    },
    "data_issue": {
        "stock": 0.20,
        "accounting": 0.19,
        "invoicing": 0.18,
        "reporting": 0.15,
        "production": 0.13,
        "CRM": 0.08,
        "integrations": 0.07,
    },
    "training_needed": {
        "general": 0.18,
        "production": 0.15,
        "invoicing": 0.14,
        "CRM": 0.13,
        "stock": 0.12,
        "reporting": 0.11,
        "accounting": 0.09,
        "user_permissions": 0.08,
    },
    "infrastructure_problem": {
        "general": 0.30,
        "integrations": 0.25,
        "reporting": 0.12,
        "user_permissions": 0.12,
        "production": 0.09,
        "invoicing": 0.07,
        "stock": 0.05,
    },
    "feature_request": {
        "reporting": 0.18,
        "production": 0.17,
        "CRM": 0.15,
        "invoicing": 0.14,
        "stock": 0.12,
        "integrations": 0.10,
        "accounting": 0.08,
        "general": 0.06,
    },
}

MODULE_WORDS = {
    "CRM": ["CRM", "contactos", "cliente", "oportunidade", "pipeline comercial"],
    "invoicing": ["fatura", "factura", "invoice", "serie de faturacao", "IVA"],
    "stock": ["stock", "armazem", "inventario", "barcode", "movimentos"],
    "production": ["producao", "ordem de fabrico", "shop floor", "planeamento", "custos"],
    "accounting": ["contabilidade", "SAF-T", "reconciliacao bancaria", "lancamentos", "IVA"],
    "reporting": ["relatorio", "dashboard", "KPI", "export Excel", "analise"],
    "user_permissions": ["permissoes", "perfil", "acessos", "role", "password"],
    "integrations": ["API", "EDI", "webservice", "integracao", "ecommerce"],
    "general": ["login", "menu", "notificacoes", "performance", "ambiente"],
}

PRIORITY_SIGNALS = {
    "critical": [
        "producao parada",
        "sem faturar",
        "cannot invoice",
        "ASAP",
        "critico",
        "todos os utilizadores bloqueados",
        "deadline hoje",
    ],
    "high": [
        "urgente",
        "cliente a espera",
        "fecho do mes",
        "afeta varios utilizadores",
        "precisamos ainda hoje",
    ],
    "medium": [
        "afeta uma equipa",
        "acontece algumas vezes",
        "precisamos esta semana",
        "impacto moderado",
    ],
    "low": [
        "sem urgencia",
        "quando possivel",
        "duvida simples",
        "melhoria para analisar",
        "pode esperar",
    ],
}

NOISE_PHRASES = [
    "nao sei se e bug ou configuracao",
    "o utilizador diz que ontem funcionava",
    "p.f. confirmar",
    "sorry pelo detalhe incompleto",
    "temos prints mas ainda nao anexei",
    "isto acontece no ambiente real",
    "maybe it is just a setting",
    "ja reiniciamos e ficou igual",
]


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    labels = list(weights)
    values = list(weights.values())
    return rng.choices(labels, weights=values, k=1)[0]


def choose_priority(rng: random.Random, ticket_type: str) -> str:
    weights = {
        "software_bug": {"medium": 0.22, "high": 0.48, "critical": 0.26, "low": 0.04},
        "client_misunderstanding": {"low": 0.45, "medium": 0.45, "high": 0.09, "critical": 0.01},
        "configuration_request": {"low": 0.12, "medium": 0.54, "high": 0.30, "critical": 0.04},
        "data_issue": {"low": 0.08, "medium": 0.40, "high": 0.42, "critical": 0.10},
        "training_needed": {"low": 0.56, "medium": 0.38, "high": 0.06, "critical": 0.00},
        "infrastructure_problem": {"low": 0.03, "medium": 0.18, "high": 0.39, "critical": 0.40},
        "feature_request": {"low": 0.62, "medium": 0.33, "high": 0.05, "critical": 0.00},
    }
    return weighted_choice(rng, weights[ticket_type])


def choose_team(ticket_type: str, priority: str, module: str) -> str:
    if ticket_type == "software_bug":
        if module in {"integrations", "general"}:
            return "technical_support"
        return "development" if priority in {"high", "critical"} else "technical_support"
    if ticket_type == "client_misunderstanding":
        return "support_level_1"
    if ticket_type == "configuration_request":
        return "technical_support"
    if ticket_type == "data_issue":
        return "technical_support"
    if ticket_type == "training_needed":
        return "training_consulting"
    if ticket_type == "infrastructure_problem":
        return "infrastructure"
    return "account_management"


def wrong_previous_classification(rng: random.Random, correct_type: str, correct_priority: str, correct_team: str) -> str:
    if rng.random() < 0.68:
        old_type = correct_type
        old_priority = correct_priority
        old_team = correct_team
    else:
        old_type = rng.choice([value for value in TICKET_TYPES if value != correct_type])
        old_priority = rng.choice([value for value in PRIORITIES if value != correct_priority])
        old_team = rng.choice([value for value in TEAMS if value != correct_team])

    if rng.random() < 0.18:
        old_team = rng.choice([value for value in TEAMS if value != correct_team])
    if rng.random() < 0.16:
        old_priority = rng.choice([value for value in PRIORITIES if value != correct_priority])
    return f"{old_type} / {old_priority} / {old_team}"


def make_subject(rng: random.Random, ticket_type: str, module: str, component: str, priority: str) -> str:
    module_word = rng.choice(MODULE_WORDS[module])
    templates = {
        "software_bug": [
            "Erro no {module_word}: {component} nao grava",
            "Bug after update in {module_word}",
            "{module_word} bloqueia ao validar {component}",
            "Exception quando usamos {component}",
        ],
        "client_misunderstanding": [
            "Duvida sobre {module_word} / {component}",
            "Como fazer no ecra de {module_word}?",
            "Utilizador nao encontra opcao em {component}",
            "Need help understanding {module_word}",
        ],
        "configuration_request": [
            "Configurar {component} em {module_word}",
            "Pedido de parametrizacao para {module_word}",
            "Alterar workflow/template de {component}",
            "Permissoes e settings para {module_word}",
        ],
        "data_issue": [
            "Dados inconsistentes em {module_word}",
            "Importacao criou registos errados no {component}",
            "Saldo/valores nao batem em {module_word}",
            "Duplicados depois de upload para {component}",
        ],
        "training_needed": [
            "Formacao para equipa em {module_word}",
            "Novos utilizadores precisam treino de {component}",
            "Sessao de esclarecimento sobre {module_word}",
            "Training request for {component}",
        ],
        "infrastructure_problem": [
            "Timeout/lentidao no acesso a {module_word}",
            "VPN/server problem affects {component}",
            "Ambiente Sistrade instavel",
            "Falha de rede ao abrir {module_word}",
        ],
        "feature_request": [
            "Melhoria desejada em {module_word}",
            "Nova opcao para {component}",
            "Feature request: {module_word}",
            "Seria possivel adicionar regra em {component}?",
        ],
    }
    subject = rng.choice(templates[ticket_type]).format(module_word=module_word, component=component)
    if priority in {"critical", "high"} and rng.random() < 0.45:
        subject = f"URGENTE - {subject}"
    return subject


def make_description(
    rng: random.Random,
    ticket_type: str,
    module: str,
    component: str,
    priority: str,
    client_name: str,
    preferred_resolver: str,
) -> str:
    module_word = rng.choice(MODULE_WORDS[module])
    signal = rng.choice(PRIORITY_SIGNALS[priority])
    noise = rng.choice(NOISE_PHRASES)

    base = {
        "software_bug": [
            "Bom dia, no cliente {client} aparece erro 500 quando tentamos usar {module_word}. O {component} fica bloqueado e nao grava os dados.",
            "After the last update, {module_word} started throwing an exception in {component}. Users repeat the same steps and it crashes.",
            "Estamos a receber stacktrace no {component}. Parece bug porque o mesmo processo funcionava ontem.",
        ],
        "client_misunderstanding": [
            "O utilizador pergunta como fazer esta operacao no {module_word}. Parece que nao encontra o menu correto para {component}.",
            "We need help understanding the flow. The client thinks something is missing, but it may be normal behavior in {module_word}.",
            "Duvida funcional: onde esta a opcao para {component}? Nao parece erro tecnico.",
        ],
        "configuration_request": [
            "Precisamos configurar regras novas no {module_word}, especialmente no {component}. E uma parametrizacao pedida pela direcao.",
            "Can you change the workflow/template for {component}? The current settings do not match the client's process.",
            "Pedido para ajustar permissoes, perfis ou parametros relacionados com {module_word}.",
        ],
        "data_issue": [
            "Os dados no {module_word} nao batem certo. Ha registos duplicados e valores estranhos depois de uma importacao CSV.",
            "Depois do upload, o {component} ficou com saldos incorretos. O cliente diz que o Excel original esta correto.",
            "A listagem mostra records em falta e totais diferentes entre ecra e relatorio.",
        ],
        "training_needed": [
            "Entraram novos utilizadores e precisam de formacao sobre {module_word}. Querem exemplos praticos para {component}.",
            "A equipa esta a usar mal o processo. Sugerimos uma sessao curta de training para evitar tickets repetidos.",
            "Precisamos de workshop para explicar o fluxo correto e boas praticas.",
        ],
        "infrastructure_problem": [
            "O sistema esta muito lento e da timeout ao abrir {module_word}. Varios utilizadores reportam falha de rede/VPN.",
            "Server seems unstable. Login works sometimes, then {component} stops responding.",
            "Desde esta manha ha problema de certificado, backup ou network e o ambiente fica inacessivel.",
        ],
        "feature_request": [
            "O cliente gostava de uma nova opcao em {module_word}. Nao e erro, e melhoria para reduzir passos manuais.",
            "Seria possivel adicionar uma regra extra no {component}? Hoje fazem isto fora do sistema.",
            "Feature request para avaliar em roadmap; o processo atual funciona mas tem muito trabalho manual.",
        ],
    }
    description = rng.choice(base[ticket_type]).format(
        client=client_name,
        module_word=module_word,
        component=component,
    )
    if preferred_resolver and rng.random() < 0.65:
        description += f" Se possivel, encaminhar para {preferred_resolver}, que ja conhece este cliente."
    if rng.random() < 0.75:
        description += f" Impacto informado: {signal}."
    if rng.random() < 0.60:
        description += f" Nota do utilizador: {noise}."
    if rng.random() < 0.25:
        description += " Mixed note: please check today if possible, the client is confused and tired."
    return description


def generate_dataset(n_tickets: int = N_TICKETS, output_path: Path = DATA_PATH) -> pd.DataFrame:
    rng = random.Random(RANDOM_STATE)
    rows = []

    for idx in range(1, n_tickets + 1):
        ticket_type = weighted_choice(rng, TYPE_WEIGHTS)
        module = weighted_choice(rng, TYPE_MODULE_WEIGHTS[ticket_type])
        component = rng.choice(MODULE_COMPONENTS[module])
        priority = choose_priority(rng, ticket_type)
        team = choose_team(ticket_type, priority, module)
        client_name, client_sector = rng.choice(CLIENTS)
        preferred_resolver = rng.choice(RESOLVERS) if rng.random() < 0.34 else ""
        urgency_signals = rng.choice(PRIORITY_SIGNALS[priority])
        subject = make_subject(rng, ticket_type, module, component, priority)
        description = make_description(
            rng,
            ticket_type,
            module,
            component,
            priority,
            client_name,
            preferred_resolver,
        )
        previous = wrong_previous_classification(rng, ticket_type, priority, team)

        rows.append(
            {
                "ticket_id": f"SIS-2026-{idx:04d}",
                "client_name": client_name,
                "client_sector": client_sector,
                "ticket_subject": subject,
                "ticket_description": description,
                "module": module,
                "component": component,
                "urgency_signals": urgency_signals,
                "previous_classification": previous,
                "correct_ticket_type": ticket_type,
                "correct_priority": priority,
                "correct_target_team": team,
                "preferred_resolver": preferred_resolver,
            }
        )

    df = pd.DataFrame(rows, columns=EXPECTED_DATA_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    df = generate_dataset()
    print(f"Generated {len(df)} synthetic tickets at {DATA_PATH}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
