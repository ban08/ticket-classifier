# Sistrade Ticket Classifier

Small university AI project for Assignment 2. The assignment asks for a low-hanging-fruit machine learning POC with artificial data, a trained model, a simple web app, and code that can be presented.

This project classifies Sistrade-style support tickets. It is not a chatbot and it does not answer customers.

The project now uses a committed AI-authored ticket corpus instead of the old template-driven synthetic generator. The goal is to feel closer to a real support queue: messy wording, mixed Portuguese/English phrasing, incomplete context, and occasional imperfect historical labels.

## What It Predicts

Given a ticket subject, description, client sector, urgency notes, previous classification, and preferred resolver, the model predicts:

- `ticket_type`
- `priority`
- `target_team`
- `module`
- `component`

## Structure

```text
.
├── app/
│   └── streamlit_app.py
├── context/
├── data/
│   └── ai_generated_sistrade_tickets.csv
├── models/
│   └── ticket_classifier.joblib
├── src/
│   ├── examples.py
│   ├── predict.py
│   ├── ticket_classifier.py
│   └── train_model.py
├── README.md
└── requirements.txt
```

## Folder Roles

- `context/`: assignment and course reference material. It is not used by the app at runtime.
- `data/`: committed AI-authored ticket data used for training and evaluation.
- `src/`: training, prediction, and shared classifier logic.
- `models/`: trained model artifact used by the CLI and app.
- `app/`: minimal Streamlit demo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train and evaluate the model:

```bash
python3 src/train_model.py
```

Run sample predictions:

```bash
python3 src/predict.py
```

Run a custom prediction:

```bash
python3 src/predict.py \
  --subject "URGENTE - erro ao emitir fatura" \
  --description "Estamos sem faturar, aparece exception no IVA e os utilizadores ficam bloqueados." \
  --client-sector "printing" \
  --urgency "sem faturar; ASAP" \
  --previous-classification "client_misunderstanding / medium / support_level_1"
```

Run debug examples:

```bash
python3 src/predict.py --evaluate-examples
```

Run the web app:

```bash
streamlit run app/streamlit_app.py
```

## Data And Model

The dataset is still artificial for the assignment, but the current version is a fixed AI-authored corpus committed in the repository. The old template script was removed.

What changed in the data approach:

1. The old `src/generate_data.py` script was removed because it produced visibly repetitive tickets and unrealistically clean class boundaries.
2. The dataset now lives in `data/ai_generated_sistrade_tickets.csv`.
3. The rows are written as realistic support-ticket examples with:
   - Portuguese and English wording mixed the way support teams often write it;
   - short, long, clean, and messy ticket styles;
   - real-business themes like invoicing, production, reporting, integrations, permissions, and accounting;
   - some ambiguous cases;
   - some wrong previous/manual classifications;
   - a small amount of intentional label noise to mimic imperfect human-labelled history.
4. The committed dataset is used for training and test splitting. The same trained model artifact is then used by the CLI and Streamlit demo.

Training uses TF-IDF text features with Logistic Regression. A Naive Bayes baseline is also trained for comparison and stored in the model artifact.

This means the project is still simple to explain:

- one committed dataset file;
- one training script;
- one shared classifier helper;
- one small app.

## Current Evaluation

The training script prints accuracy, precision, recall, and weighted F1 for each predicted label. These scores come from a fixed train/test split on the committed AI-authored dataset.

The exact numbers may change if the committed dataset changes. Unlike the old version, extremely high scores are no longer the target; some ambiguity and small label noise are intentional. Treat the results as a realistic classroom POC, not as proof of production performance.

## Conservative Classification

The classifier now keeps a slight business-risk bias during prediction.

What that means here:

1. The base model still predicts normally from text.
2. A small post-processing step only nudges close calls.
3. The nudge prefers the safer business mistake when the model is uncertain.

Priority examples:

- Safer mistake: classify a borderline medium/high incident as `high`.
- More dangerous mistake: classify a real invoicing outage or login outage as `low`.

Type examples:

- Safer mistake: lean a little toward `software_bug`, `data_issue`, or `infrastructure_problem` when the ticket contains strong incident language.
- More dangerous mistake: dismiss a production-impacting incident as `client_misunderstanding` or `training_needed` too early.

Routing examples:

- Safer mistake: in a close call, lean slightly toward a technical queue for risky incidents.
- More dangerous mistake: leave a real outage-looking ticket in a low-risk front-line bucket.

Module examples:

- Safer mistake: when text clearly hints at invoicing/accounting/integrations and the model is split, give a small boost to those business-critical modules.
- More dangerous mistake: route a finance-impacting issue into a softer adjacent module because the wording was incomplete.

This is intentionally conservative, not extreme. The model is not designed to label everything urgent. It only adds a mild bias where missing a real incident is more expensive than escalating a false alarm.

## Assignment Fit

The project matches `context/assign2.html`:

- identifies a practical real-world ML problem;
- uses artificial data;
- trains a predictive model;
- provides a small web app demonstrating the model;
- keeps implementation simple enough to explain in a presentation.

## Limitations

- The data is synthetic, not real Sistrade data.
- The dataset is more realistic than the old template version, but real tickets would still contain more customer-specific shorthand, attachments, and internal history.
- `component` can be harder than broader labels because components are detailed and similar.
- Some intentional label noise was kept to make the training set feel more like human-labelled history, so the dataset is not perfectly clean.
- `target_team` is still simplified compared with a real multi-team support operation.
- The model suggests labels for human review; it should not automatically route or answer tickets in production.
