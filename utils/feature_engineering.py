from config import TRAINING_THRESHOLD


def engineer_features(raw: dict) -> dict:
    """
    raw must contain: type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, step

    Returns the 10 engineered features (not including one-hot type columns) —
    computed with the exact formulas used during model training.
    """
    amount = raw["amount"]
    old_org = raw["oldbalanceOrg"]
    new_org = raw["newbalanceOrig"]
    old_dest = raw["oldbalanceDest"]
    new_dest = raw["newbalanceDest"]
    step = raw["step"]

    # Simulated hour and day (PaySim's `step` counts hours continuously across
    # the whole ~31-day simulation, 1-743 — it does NOT reset every 24 hours)
    hour = step % 24
    day = (step // 24) + 1

    # Balance change
    sender_balance_change = old_org - new_org
    receiver_balance_change = new_dest - old_dest

    # Amount-to-balance ratio
    amount_balance_ratio = amount / (old_org + 1)

    # Account emptied flag
    account_emptied = int(new_org == 0)

    # Large transaction flag — uses the FIXED training threshold, not live data
    large_transaction = int(amount > TRAINING_THRESHOLD)

    # Zero balance flags
    sender_zero_balance = int(old_org == 0)
    receiver_zero_balance = int(old_dest == 0)

    # Receiver balance didn't move despite a nonzero amount — fraud in over
    # 97% of TRANSFER cases in the training data (added after analyzing why
    # the original model missed this exact pattern; see project notes).
    dest_not_credited = int(receiver_balance_change == 0 and amount > 0)

    return {
        "isFlaggedFraud": 0,  # PaySim system flag, not user-derivable — default 0
        "sender_balance_change": sender_balance_change,
        "receiver_balance_change": receiver_balance_change,
        "hour": hour,
        "day": day,
        "amount_balance_ratio": amount_balance_ratio,
        "account_emptied": account_emptied,
        "large_transaction": large_transaction,
        "sender_zero_balance": sender_zero_balance,
        "receiver_zero_balance": receiver_zero_balance,
        "dest_not_credited": dest_not_credited,
    }
