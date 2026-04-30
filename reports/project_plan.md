# Project Plan

## Scope

Build a university-scale proof of concept that demonstrates how machine learning could support Sistrade ticket triage.

Included:

- identify and describe the business problem;
- generate artificial Sistrade-like support tickets;
- train local deterministic ML classifiers;
- evaluate the classifiers with standard metrics;
- build a simple Streamlit app for demonstration;
- provide README, reports, and presentation outline.

Excluded:

- chatbot answering;
- OpenAI API or external LLMs;
- real customer data;
- production deployment;
- integration with Sistrade systems;
- automatic customer communication.

## Workflow

1. Extract official assignment requirements from the Moodle HTML file.
2. Define a low-hanging-fruit classification task instead of a broad chatbot.
3. Generate synthetic tickets with realistic ERP/support language.
4. Train one classifier for each output label.
5. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrices.
6. Save the trained pipeline.
7. Build a Streamlit app to enter a new ticket and inspect predictions.
8. Add tests for deterministic behavior and plausible outputs.
9. Document the project honestly, including synthetic-data limitations.

## Deliverables

- `data/synthetic_sistrade_tickets.csv`
- `models/ticket_classifier.joblib`
- `app/streamlit_app.py`
- `src/generate_data.py`
- `src/train_model.py`
- `src/predict.py`
- `tests/test_predictions.py`
- README and reports in `reports/`

## Success Criteria

- The project can be run locally with the documented commands.
- The model produces deterministic predictions.
- The app demonstrates ticket classification clearly.
- Evaluation metrics are reported.
- The documentation explains what the project is and is not.
