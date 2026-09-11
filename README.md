# ticket-classifier

Reads a customer support ticket and predicts how it should be routed — its type, priority, team, and the product area it touches.

## What it does

Takes the free text of a support ticket and predicts five things: `ticket_type`, `priority`, `target_team`, `module` and `component`. It ships with the dataset, the training code, a command-line tool for predictions, and a small Streamlit interface for trying it interactively. The dataset is synthetic — modelled on the kind of tickets an industrial-software support desk sees, with invented client names.

## Stack

Python, scikit-learn (the trained model is saved as a `joblib` file), and Streamlit for the UI.

## How to run

```bash
pip install -r requirements.txt
python src/train_model.py         # train the classifier
python src/predict.py "ticket text here"   # predict from the CLI
streamlit run app/streamlit_app.py         # interactive UI
```

## What I built

Group project for the Artificial Intelligence course (2025/26), built with a teammate whose account hosts the original repository. The work was done together; it is not separated cleanly in the git history, so I do not want to overstate a specific split here.

## What I would do differently

Report a proper evaluation — a train/test split with per-label precision and recall, and a confusion matrix for the priority field — rather than relying on the trained model alone, and handle the class imbalance between common and rare ticket types.
