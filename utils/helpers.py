from config import TRAINING_THRESHOLD

BALANCE_TOLERANCE = 1.0  # ₹1 tolerance for floating-point rounding


def validate_transaction(data: dict) -> list:
    """
    Business-rule sanity checks, separate from the ML model. Runs before the
    model ever sees the input and catches obviously bad/inconsistent data.

    Deliberately does NOT check receiver-side consistency: PaySim's own
    training data has many legitimate transactions (CASH_IN, PAYMENT to
    merchants) where the receiver's balance doesn't move by the full amount,
    and the model relies on that exact pattern (dest_not_credited) as a real
    fraud signal — enforcing it here would reject valid inputs the model was
    trained to handle.

    Returns: list of error strings (empty list = valid)
    """
    errors = []

    txn_type = data["type"]
    amount = data["amount"]
    old_org = data["oldbalanceOrg"]
    new_org = data["newbalanceOrig"]
    old_dest = data["oldbalanceDest"]
    new_dest = data["newbalanceDest"]
    step = data["step"]

    # ── Basic numeric validation (all fields, non-negative) ────
    if amount < 0:
        errors.append("Amount cannot be negative.")
    if old_org < 0:
        errors.append("Sender's balance before cannot be negative.")
    if new_org < 0:
        errors.append("Sender's balance after cannot be negative.")
    if old_dest < 0:
        errors.append("Receiver's balance before cannot be negative.")
    if new_dest < 0:
        errors.append("Receiver's balance after cannot be negative.")
    if step < 0:
        errors.append("Time step cannot be negative.")

    # ── Sender-side consistency check only ─────────────────────
    sender_change = old_org - new_org

    if txn_type == "CASH_IN":
        # Sender is receiving money, so their balance should increase by ~amount.
        if abs(sender_change - (-amount)) > BALANCE_TOLERANCE:
            errors.append(
                f"For a CASH_IN, the sender's balance should *increase* by ~₹{amount:,.0f}, "
                f"but it actually changed by ₹{-sender_change:,.0f}."
            )
    else:
        if new_org > old_org:
            errors.append(
                f"Sender's balance increased, but a {txn_type} transaction should only decrease it."
            )
        if abs(sender_change - amount) > BALANCE_TOLERANCE:
            errors.append(
                f"Sender's balance changed by ₹{sender_change:,.0f}, but the stated amount is "
                f"₹{amount:,.0f} — these should match for a {txn_type} transaction."
            )

    return errors


def get_risk_factors(raw: dict, engineered: dict) -> list:
    """
    Rule-based, human-readable risk factors for the Risk Factor Breakdown card —
    interpretable if/else logic layered on top of the real prediction, using the
    same engineered features the model used. Not a second model.

    Returns: list of (active: bool, active_text: str, inactive_text: str) tuples.
    """
    amount = raw["amount"]

    return [
        (
            bool(engineered["dest_not_credited"]),
            f"<b>Receiver balance did not change</b> despite a ₹{amount:,.0f} transfer — "
            "this exact pattern is fraud in over 97% of similar cases in the training data.",
            "Receiver balance updated consistently with the transaction amount.",
        ),
        (
            bool(engineered["account_emptied"]),
            "<b>Sender account was fully drained</b> to zero — a classic fraud signature "
            "(drain-then-cashout).",
            "Sender retains a remaining balance after the transaction.",
        ),
        (
            bool(engineered["large_transaction"]),
            f"<b>Unusually large amount</b> — above the 95th percentile (₹{TRAINING_THRESHOLD:,.0f}) "
            "seen in the training data.",
            "Amount is within the typical range seen in the training data.",
        ),
        (
            bool(engineered["sender_zero_balance"]),
            "Sender account had a <b>zero starting balance</b>, which is atypical for an "
            "active account.",
            "Sender account had a non-zero starting balance.",
        ),
    ]
