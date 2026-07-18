# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Model & Encoder Paths ──────────────────────────────────────
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "onehot_encoder.pkl"

# ── Feature Engineering Constants ──────────────────────────────
# 95th percentile of 'amount' from the training dataset — fixed at training time.
# Recalculating this from live user input would make the threshold meaningless,
# since the model learned what "large" means relative to the training distribution.
TRAINING_THRESHOLD = 518634.19649999996

# ── Prediction Logging ──────────────────────────────────────────
PREDICTION_LOG_PATH = BASE_DIR / "prediction_logs" / "predictions.csv"
