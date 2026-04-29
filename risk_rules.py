from __future__ import annotations

import pandas as pd


def build_model_frame(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(accounts, on="account_id", how="left")

    df["is_large_amount"] = (df["amount_usd"] >= 1000).astype(int)
    df["login_pressure"] = pd.cut(
        df["failed_logins_24h"],
        bins=[-1, 0, 2, 100],
        labels=["none", "low", "high"]
    )

    # Flag accounts whose current velocity is unusually high relative to their
    # own history. A $200 charge from an account that normally makes 1 txn/day
    # at 8 txns in 24h is far more suspicious than a normally active account.
    if "avg_daily_velocity" in df.columns:
        df["velocity_spike"] = (
            df["velocity_24h"] > df["avg_daily_velocity"] * 3
        ).astype(int)
    else:
        df["velocity_spike"] = 0

    return df
