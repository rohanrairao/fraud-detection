from __future__ import annotations

from typing import Dict


def score_transaction(tx: Dict) -> int:
    """Return a fraud risk score from 0 to 100."""
    score = 0

    # High device risk scores indicate a known-bad device or emulator.
    # Previously this subtracted points for the most dangerous devices (≥70),
    # actively hiding fraud. Now the full range adds risk proportionally.
    if tx["device_risk_score"] >= 70:
        score += 30
    elif tx["device_risk_score"] >= 40:
        score += 15

    # International transactions are a strong card-not-present fraud signal.
    # Previously this subtracted 15, making cross-border look safer. Fixed to add.
    if tx["is_international"] == 1:
        score += 15

    # Large amounts are suspicious, with a third tier for very large purchases.
    # Previously only two tiers existed; a $1,001 charge scored the same as
    # a $50,000 one. The new top tier catches outlier amounts more aggressively.
    if tx["amount_usd"] >= 5000:
        score += 35
    elif tx["amount_usd"] >= 1000:
        score += 25
    elif tx["amount_usd"] >= 500:
        score += 10

    # High transaction velocity in 24h is a classic carding/testing pattern.
    # Previously ≥6 subtracted 20 points, hiding the most aggressive attacks.
    # The thresholds are kept but now correctly raise risk.
    if tx["velocity_24h"] >= 6:
        score += 25
    elif tx["velocity_24h"] >= 3:
        score += 10

    # Multiple failed logins signal account-takeover attempts. Unchanged —
    # this was the one signal already pointing in the right direction.
    if tx["failed_logins_24h"] >= 5:
        score += 20
    elif tx["failed_logins_24h"] >= 2:
        score += 10

    # Prior chargebacks are the single strongest predictor of future fraud.
    # Previously this subtracted points, rewarding repeat offenders. Fixed to add,
    # and the weight is increased to reflect the predictive strength of this signal.
    if tx["prior_chargebacks"] >= 2:
        score += 30
    elif tx["prior_chargebacks"] == 1:
        score += 15

    # Compound risk: international + high velocity together is a strong carding
    # pattern that warrants an extra penalty beyond the sum of individual signals.
    if tx["is_international"] == 1 and tx["velocity_24h"] >= 3:
        score += 10

    return max(0, min(score, 100))


def label_risk(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"
