"""Train and evaluate deterministic ticket classification models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ticket_classifier import (
    DATA_PATH,
    EXPECTED_DATA_COLUMNS,
    MODEL_PATH,
    OUTPUT_LABELS,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_ticket_text,
    KEYWORD_RULES,
)


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=14000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_nb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=14000,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    labels = sorted(y_true.unique())
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": labels,
    }


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_DATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) < 500:
        raise ValueError("Dataset is too small for this POC. Expected at least 500 rows.")


def project_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_models(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run `python src/generate_data.py` first.")

    df = pd.read_csv(data_path).fillna("")
    validate_dataset(df)
    x = df.apply(build_ticket_text, axis=1)

    train_index, test_index = train_test_split(
        df.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["correct_ticket_type"],
    )
    x_train = x.loc[train_index]
    x_test = x.loc[test_index]

    models = {}
    metrics = {}
    baselines = {}

    for output_name, column_name in OUTPUT_LABELS.items():
        y_train = df.loc[train_index, column_name]
        y_test = df.loc[test_index, column_name]

        baseline = make_nb_pipeline()
        baseline.fit(x_train, y_train)
        baseline_pred = baseline.predict(x_test)
        baselines[output_name] = evaluate_predictions(y_test, baseline_pred)

        model = make_logistic_pipeline()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        models[output_name] = model
        metrics[output_name] = evaluate_predictions(y_test, prediction)

    artifact = {
        "project": "sistrade_ticket_classification_poc",
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset_path": project_relative_path(data_path),
        "dataset_rows": int(len(df)),
        "test_size": 0.20,
        "label_targets": OUTPUT_LABELS,
        "algorithm": "TF-IDF + LogisticRegression",
        "baseline_algorithm": "TF-IDF + MultinomialNB",
        "models": models,
        "metrics": metrics,
        "baselines": baselines,
        "keyword_rules": KEYWORD_RULES,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    return artifact


def print_metrics(artifact: dict) -> None:
    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Dataset rows: {artifact['dataset_rows']}")
    print("\nFinal model metrics (weighted averages):")
    for output_name, values in artifact["metrics"].items():
        print(
            f"- {output_name}: accuracy={values['accuracy']:.4f}, "
            f"precision={values['precision_weighted']:.4f}, "
            f"recall={values['recall_weighted']:.4f}, f1={values['f1_weighted']:.4f}"
        )
    print("\nBaseline comparison is stored inside the model artifact.")


def main() -> None:
    artifact = train_models()
    print_metrics(artifact)


if __name__ == "__main__":
    main()
