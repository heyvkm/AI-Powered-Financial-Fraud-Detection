import csv
from datetime import datetime

from config import PREDICTION_LOG_PATH

FIELDS = [
    "timestamp", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "step",
    "prediction", "label", "fraud_probability", "risk_level",
]


def log_prediction(raw: dict, result: dict) -> None:
    """Append one row per completed prediction to prediction_logs/predictions.csv.

    Only called for validated, successfully-predicted transactions — inputs
    rejected by validate_transaction() never reach the model, so they're not
    logged as predictions.
    """
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not PREDICTION_LOG_PATH.exists()

    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": raw["type"],
            "amount": raw["amount"],
            "oldbalanceOrg": raw["oldbalanceOrg"],
            "newbalanceOrig": raw["newbalanceOrig"],
            "oldbalanceDest": raw["oldbalanceDest"],
            "newbalanceDest": raw["newbalanceDest"],
            "step": raw["step"],
            "prediction": result["prediction"],
            "label": result["label"],
            "fraud_probability": round(result["fraud_probability"], 4),
            "risk_level": result["risk_level"],
        })
