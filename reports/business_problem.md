# Business Problem

## Company Context

Sistrade provides ERP software for industrial companies such as printing, packaging, labels, and related manufacturing sectors. Customers contact support when they face problems with modules such as invoicing, stock, production, CRM, accounting, reporting, permissions, and integrations.

Support tickets arrive with different levels of detail. Some are urgent and technical, while others are misunderstandings, configuration requests, training needs, data corrections, or improvement ideas.

## Current Pain

Each ticket has to be read, classified, prioritized, and assigned by support staff. This manual work is repetitive and can be affected by pressure, incomplete descriptions, and staff fatigue.

Wrong classifications can happen in several ways:

- a client misunderstanding is sent to a technical team;
- a software bug is treated as a simple user doubt;
- a critical invoicing problem is marked as low priority;
- a low-risk request is escalated unnecessarily;
- the ticket is routed to the wrong department or resolver.

## Why Wrong Classification Costs Time And Money

Misclassification creates extra handoffs. A team may inspect a ticket, realize it does not belong to them, add notes, and send it elsewhere. This wastes support and technical staff time.

It can also delay urgent work. If a customer cannot invoice or production is blocked, slow routing can create commercial pressure and customer dissatisfaction.

The cost is not only the time spent on one ticket. Repeated wrong routing increases queue noise, interrupts specialists, and makes prioritization less reliable.

## Why This Is Low-Hanging Fruit For ML

Ticket classification is a realistic ML task because historical tickets usually contain text, metadata, and final classifications. Even a simple classifier can learn common patterns such as:

- invoice, IVA, SAF-T, and credit note language;
- urgency words such as "ASAP", "bloqueado", and "sem faturar";
- configuration words such as "perfil", "template", and "workflow";
- infrastructure words such as "VPN", "server", and "timeout".

This POC keeps the risk low. It does not answer clients directly and does not replace staff. It only suggests classifications for human review.
