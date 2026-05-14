# Sistrade Ticket Classifier

Small university AI project for Assignment 2. The assignment asks for a low-hanging-fruit machine learning POC with artificial data, a trained model, a simple web app, and code that can be presented.

This project classifies Sistrade-style support tickets. It is not a chatbot and it does not answer customers.

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
│   └── synthetic_sistrade_tickets.csv
├── models/
│   └── ticket_classifier.joblib
├── src/
│   ├── examples.py
│   ├── generate_data.py
│   ├── predict.py
│   ├── ticket_classifier.py
│   └── train_model.py
├── README.md
└── requirements.txt
```

## Folder Roles

- `context/`: assignment and course reference material. It is not used by the app at runtime.
- `data/`: synthetic ticket data used for training and evaluation.
- `src/`: data generation, training, prediction, and shared classifier logic.
- `models/`: trained model artifact used by the CLI and app.
- `app/`: minimal Streamlit demo.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Generate synthetic data:

```bash
python src/generate_data.py
```

Train and evaluate the model:

```bash
python src/train_model.py
```

Run sample predictions:

```bash
python src/predict.py
```

Run a custom prediction:

```bash
python src/predict.py \
  --subject "URGENTE - erro ao emitir fatura" \
  --description "Estamos sem faturar, aparece exception no IVA e os utilizadores ficam bloqueados." \
  --client-sector "printing" \
  --urgency "sem faturar; ASAP" \
  --previous-classification "client_misunderstanding / medium / support_level_1"
```

Run debug examples:

```bash
python src/predict.py --evaluate-examples
```

Run the web app:

```bash
streamlit run app/streamlit_app.py
```

## Data And Model

The dataset is artificial and generated locally. It includes realistic ticket variation for the POC:

- email-like support messages;
- Portuguese and English wording;
- ERP vocabulary for invoicing, stock, production, CRM, accounting, reporting, permissions, integrations, and general issues;
- urgent and non-urgent requests;
- unclear or noisy descriptions;
- sometimes wrong previous classifications.

Training uses TF-IDF text features with Logistic Regression. A Naive Bayes baseline is also trained for comparison and stored in the model artifact.

## Current Evaluation

The training script prints accuracy, precision, recall, and weighted F1 for each predicted label. These scores come from a fixed train/test split on synthetic data.

The exact numbers may change if the synthetic generator changes. Treat them as proof that the workflow works, not proof of real-world performance.

## Assignment Fit

The project matches `context/assign2.html`:

- identifies a practical real-world ML problem;
- uses artificial data;
- trains a predictive model;
- provides a small web app demonstrating the model;
- keeps implementation simple enough to explain in a presentation.

## Limitations

- The data is synthetic, not real Sistrade data.
- Real tickets would contain more missing context, attachments, spelling variation, and customer-specific language.
- `component` can be harder than broader labels because components are detailed and similar.
- `target_team` is simplified with triage rules in the synthetic data.
- The model suggests labels for human review; it should not automatically route or answer tickets in production.
