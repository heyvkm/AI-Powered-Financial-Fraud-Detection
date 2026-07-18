import pandas as pd

from utils.loader import load_model, load_encoder
from utils.feature_engineering import engineer_features

ALL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def encode_type(txn_type: str, encoder) -> dict:
    """One-hot encode the transaction type using the saved encoder."""
    input_df = pd.DataFrame([[txn_type]], columns=["type"])
    encoded = encoder.transform(input_df)
    columns = encoder.get_feature_names_out(["type"])
    return dict(zip(columns, encoded[0]))


def get_risk_level(probability: float) -> str:
    if probability < 0.20:
        return "LOW"
    elif probability < 0.50:
        return "MEDIUM"
    elif probability < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"


def predict(raw: dict) -> dict:
    """
    raw: {
        "type": str, "amount": float, "oldbalanceOrg": float,
        "newbalanceOrig": float, "oldbalanceDest": float,
        "newbalanceDest": float, "step": int
    }
    """
    model = load_model()
    encoder = load_encoder()

    # 1. Engineered features (includes dest_not_credited)
    engineered = engineer_features(raw)

    # 2. One-hot encode type
    type_encoded = encode_type(raw["type"], encoder)

    # 3. Combine all features into one dict
    full_features = {
        "step": raw["step"],
        "amount": raw["amount"],
        "oldbalanceOrg": raw["oldbalanceOrg"],
        "newbalanceOrig": raw["newbalanceOrig"],
        "oldbalanceDest": raw["oldbalanceDest"],
        "newbalanceDest": raw["newbalanceDest"],
        **engineered,
        **type_encoded,
    }

    # 4. Arrange in exact order the model expects — pulled live from the model
    #    object, never hardcoded, so a retrain with a different column order
    #    doesn't silently break this.
    ordered_columns = list(model.feature_names_in_)
    input_df = pd.DataFrame([full_features])[ordered_columns]

    # 5. Predict
    prediction = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]
    fraud_probability = float(proba[1])
    confidence = float(max(proba))

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": fraud_probability,
        "risk_level": get_risk_level(fraud_probability),
        "confidence": confidence,
        "engineered": engineered,
        "input_df": input_df,
    }
