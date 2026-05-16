# Sistrade Ticket Classifier

Ticket classification project for a Sistrade-style support operation. The model reads ticket text and predicts:

- `ticket_type`
- `priority`
- `target_team`
- `module`
- `component`

The repository includes the dataset, training code, command-line interface, and a small Streamlit interface.

## Structure

```text
.
├── app/
│   └── streamlit_app.py
├── context/
├── data/
│   └── tickets_dataset.csv
├── models/
│   └── ticket_classifier.joblib
├── src/
│   ├── predict.py
│   ├── sample_tickets.py
│   ├── ticket_classifier.py
│   └── train_model.py
├── README.md
└── requirements.txt
```

## Repository Layout

- `data/` holds the ticket dataset used for training and evaluation.
- `src/` contains model training, prediction, and shared classifier logic.
- `app/` contains the Streamlit interface.
- `models/` stores the trained artifact used by the CLI and the app.
- `context/` contains assignment material and lecture references.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train the model:

```bash
python3 src/train_model.py
```

Run predictions from the command line:

```bash
python3 src/predict.py
```

Run the built-in checks:

```bash
python3 src/predict.py --check-samples
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

Start the interface:

```bash
streamlit run app/streamlit_app.py
```

## Data

`data/tickets_dataset.csv` contains business-style support tickets written with varied tone and quality:

- short and long requests
- Portuguese and English mixed where it fits the support context
- incomplete descriptions
- ambiguous cases near class boundaries
- previous classifications that are sometimes inconsistent
- a small amount of label noise, which is common in manually triaged histories

The dataset is used both for model fitting and for the train/test split reported by the training script.

## Model

Training uses TF-IDF features with Logistic Regression. A Multinomial Naive Bayes baseline is trained alongside it and stored in the model artifact for comparison.

The classifier keeps a small conservative bias on close decisions:

- urgent operational or financial incidents should not drift toward low priority
- incident-like wording should not be dismissed too quickly as training or misunderstanding
- finance, permissions, and integration-related wording gets slightly more weight when the model is split across nearby options

The adjustment is limited to close calls. It is meant to reduce costly misses, not to inflate everything to the highest severity.

## Evaluation

`src/train_model.py` prints accuracy, precision, recall, and weighted F1 for each target on a fixed train/test split.

These scores describe performance on the repository dataset only. They are useful for comparing revisions of the project, not for claiming production accuracy.

## Notes

- The project uses synthetic tickets rather than customer data.
- `component` is the hardest field because many labels are narrow and text overlaps across modules.
- `target_team` is simplified compared with a real support organization.
- Predictions are intended to support triage, not replace human review.
