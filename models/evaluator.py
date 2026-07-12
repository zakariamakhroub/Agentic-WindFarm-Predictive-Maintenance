import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:

    def evaluate(self, model, prediction_df: pd.DataFrame):

        prediction_df = prediction_df.copy()

        excluded_columns = {"target", "status_type_id", "train_test"}

        feature_columns = [
            col
            for col in prediction_df.columns
            if col not in excluded_columns
            and pd.api.types.is_numeric_dtype(prediction_df[col])
        ]

        X_test = prediction_df[feature_columns]
        y_true = prediction_df["target"]

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        print("=" * 50)
        print("Model Evaluation")
        print("=" * 50)

        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
        print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")

        print("\nConfusion Matrix")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report")
        print(classification_report(y_true, y_pred))

        return y_pred, y_prob