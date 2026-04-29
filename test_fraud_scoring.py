"""
Fraud detection test suite.

Note on file layout: due to scrambled filenames in this repo,
  score_transaction / label_risk  live in test_risk_rules.py
  build_model_frame               lives in risk_rules.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from risk_rules import build_model_frame
from test_risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tx(**overrides) -> dict:
    """Minimal zero-risk transaction; override any field."""
    base = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    base.update(overrides)
    return base


def make_frames(tx_rows: list[dict], acct_rows: list[dict]):
    return pd.DataFrame(tx_rows), pd.DataFrame(acct_rows)


# ---------------------------------------------------------------------------
# label_risk — exact boundary behaviour
# ---------------------------------------------------------------------------

class TestLabelRisk:
    def test_zero_is_low(self):
        assert label_risk(0) == "low"

    def test_29_is_low(self):
        assert label_risk(29) == "low"

    def test_30_is_medium(self):
        assert label_risk(30) == "medium"

    def test_59_is_medium(self):
        assert label_risk(59) == "medium"

    def test_60_is_high(self):
        assert label_risk(60) == "high"

    def test_100_is_high(self):
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — device_risk_score
# ---------------------------------------------------------------------------

class TestDeviceRiskScore:
    def test_score_0_adds_nothing(self):
        assert score_transaction(tx(device_risk_score=0)) == 0

    def test_score_39_adds_nothing(self):
        assert score_transaction(tx(device_risk_score=39)) == 0

    def test_score_40_adds_15(self):
        assert score_transaction(tx(device_risk_score=40)) == 15

    def test_score_69_adds_15(self):
        assert score_transaction(tx(device_risk_score=69)) == 15

    def test_score_70_adds_30(self):
        assert score_transaction(tx(device_risk_score=70)) == 30

    def test_score_100_adds_30(self):
        assert score_transaction(tx(device_risk_score=100)) == 30

    def test_boundary_39_vs_40(self):
        assert score_transaction(tx(device_risk_score=39)) < score_transaction(tx(device_risk_score=40))

    def test_boundary_69_vs_70(self):
        assert score_transaction(tx(device_risk_score=69)) < score_transaction(tx(device_risk_score=70))


# ---------------------------------------------------------------------------
# score_transaction — is_international
# ---------------------------------------------------------------------------

class TestIsInternational:
    def test_domestic_adds_nothing(self):
        assert score_transaction(tx(is_international=0)) == 0

    def test_international_adds_15(self):
        assert score_transaction(tx(is_international=1)) == 15

    def test_international_raises_score_vs_domestic(self):
        assert score_transaction(tx(is_international=1)) > score_transaction(tx(is_international=0))


# ---------------------------------------------------------------------------
# score_transaction — amount_usd
# ---------------------------------------------------------------------------

class TestAmountUsd:
    def test_below_500_adds_nothing(self):
        assert score_transaction(tx(amount_usd=0)) == 0
        assert score_transaction(tx(amount_usd=499)) == 0

    def test_500_adds_10(self):
        assert score_transaction(tx(amount_usd=500)) == 10

    def test_999_adds_10(self):
        assert score_transaction(tx(amount_usd=999)) == 10

    def test_1000_adds_25(self):
        assert score_transaction(tx(amount_usd=1000)) == 25

    def test_4999_adds_25(self):
        assert score_transaction(tx(amount_usd=4999)) == 25

    def test_5000_adds_35(self):
        assert score_transaction(tx(amount_usd=5000)) == 35

    def test_very_large_amount_adds_35(self):
        assert score_transaction(tx(amount_usd=100_000)) == 35

    def test_boundary_499_vs_500(self):
        assert score_transaction(tx(amount_usd=499)) < score_transaction(tx(amount_usd=500))

    def test_boundary_999_vs_1000(self):
        assert score_transaction(tx(amount_usd=999)) < score_transaction(tx(amount_usd=1000))

    def test_boundary_4999_vs_5000(self):
        assert score_transaction(tx(amount_usd=4999)) < score_transaction(tx(amount_usd=5000))


# ---------------------------------------------------------------------------
# score_transaction — velocity_24h
# ---------------------------------------------------------------------------

class TestVelocity24h:
    def test_velocity_1_adds_nothing(self):
        assert score_transaction(tx(velocity_24h=1)) == 0

    def test_velocity_2_adds_nothing(self):
        assert score_transaction(tx(velocity_24h=2)) == 0

    def test_velocity_3_adds_10(self):
        assert score_transaction(tx(velocity_24h=3)) == 10

    def test_velocity_5_adds_10(self):
        assert score_transaction(tx(velocity_24h=5)) == 10

    def test_velocity_6_adds_25(self):
        assert score_transaction(tx(velocity_24h=6)) == 25

    def test_velocity_50_adds_25(self):
        assert score_transaction(tx(velocity_24h=50)) == 25

    def test_boundary_2_vs_3(self):
        assert score_transaction(tx(velocity_24h=2)) < score_transaction(tx(velocity_24h=3))

    def test_boundary_5_vs_6(self):
        assert score_transaction(tx(velocity_24h=5)) < score_transaction(tx(velocity_24h=6))


# ---------------------------------------------------------------------------
# score_transaction — failed_logins_24h
# ---------------------------------------------------------------------------

class TestFailedLogins24h:
    def test_0_logins_adds_nothing(self):
        assert score_transaction(tx(failed_logins_24h=0)) == 0

    def test_1_login_adds_nothing(self):
        assert score_transaction(tx(failed_logins_24h=1)) == 0

    def test_2_logins_adds_10(self):
        assert score_transaction(tx(failed_logins_24h=2)) == 10

    def test_4_logins_adds_10(self):
        assert score_transaction(tx(failed_logins_24h=4)) == 10

    def test_5_logins_adds_20(self):
        assert score_transaction(tx(failed_logins_24h=5)) == 20

    def test_many_logins_adds_20(self):
        assert score_transaction(tx(failed_logins_24h=20)) == 20

    def test_boundary_1_vs_2(self):
        assert score_transaction(tx(failed_logins_24h=1)) < score_transaction(tx(failed_logins_24h=2))

    def test_boundary_4_vs_5(self):
        assert score_transaction(tx(failed_logins_24h=4)) < score_transaction(tx(failed_logins_24h=5))


# ---------------------------------------------------------------------------
# score_transaction — prior_chargebacks
# ---------------------------------------------------------------------------

class TestPriorChargebacks:
    def test_0_chargebacks_adds_nothing(self):
        assert score_transaction(tx(prior_chargebacks=0)) == 0

    def test_1_chargeback_adds_15(self):
        assert score_transaction(tx(prior_chargebacks=1)) == 15

    def test_2_chargebacks_adds_30(self):
        assert score_transaction(tx(prior_chargebacks=2)) == 30

    def test_many_chargebacks_adds_30(self):
        assert score_transaction(tx(prior_chargebacks=10)) == 30

    def test_boundary_0_vs_1(self):
        assert score_transaction(tx(prior_chargebacks=0)) < score_transaction(tx(prior_chargebacks=1))

    def test_boundary_1_vs_2(self):
        assert score_transaction(tx(prior_chargebacks=1)) < score_transaction(tx(prior_chargebacks=2))


# ---------------------------------------------------------------------------
# score_transaction — compound international + velocity signal
# ---------------------------------------------------------------------------

class TestCompoundSignal:
    def test_intl_plus_moderate_velocity_exact_score(self):
        # 15 (intl) + 10 (vel 3–5) + 10 (compound) = 35
        assert score_transaction(tx(is_international=1, velocity_24h=3)) == 35

    def test_intl_plus_high_velocity_exact_score(self):
        # 15 (intl) + 25 (vel ≥6) + 10 (compound) = 50
        assert score_transaction(tx(is_international=1, velocity_24h=6)) == 50

    def test_no_compound_without_international(self):
        # velocity alone at 4: just +10, no compound bonus
        assert score_transaction(tx(is_international=0, velocity_24h=4)) == 10

    def test_no_compound_with_velocity_below_3(self):
        # international but velocity=2: just +15 from intl, no compound
        assert score_transaction(tx(is_international=1, velocity_24h=2)) == 15

    def test_combined_exceeds_sum_of_independent_signals(self):
        # intl_only=15, vel_only=10, combined=35 > 25
        intl_only = score_transaction(tx(is_international=1, velocity_24h=1))
        vel_only = score_transaction(tx(is_international=0, velocity_24h=4))
        combined = score_transaction(tx(is_international=1, velocity_24h=4))
        assert combined > intl_only + vel_only


# ---------------------------------------------------------------------------
# score_transaction — score bounds
# ---------------------------------------------------------------------------

class TestScoreBounds:
    def test_minimum_score_is_zero(self):
        assert score_transaction(tx()) == 0

    def test_all_signals_at_max_clamps_to_100(self):
        assert score_transaction(tx(
            device_risk_score=100, is_international=1, amount_usd=99_999,
            velocity_24h=99, failed_logins_24h=99, prior_chargebacks=99,
        )) == 100

    def test_score_never_exceeds_100(self):
        result = score_transaction(tx(
            device_risk_score=100, is_international=1, amount_usd=99_999,
            velocity_24h=99, failed_logins_24h=99, prior_chargebacks=99,
        ))
        assert result <= 100

    def test_score_never_below_zero(self):
        assert score_transaction(tx(
            device_risk_score=0, is_international=0, amount_usd=0,
            velocity_24h=0, failed_logins_24h=0, prior_chargebacks=0,
        )) >= 0


# ---------------------------------------------------------------------------
# Known fraud patterns — scenarios that must score high
# ---------------------------------------------------------------------------

class TestKnownFraudPatterns:
    def test_carding_attack(self):
        """Rapid small test charges from a suspicious device abroad.
        device=75 (+30) + intl (+15) + vel=10 (+25) + compound (+10) = 80
        """
        assert score_transaction(tx(
            device_risk_score=75,
            is_international=1,
            amount_usd=15,
            velocity_24h=10,
            failed_logins_24h=0,
            prior_chargebacks=0,
        )) == 80

    def test_account_takeover_large_withdrawal(self):
        """Brute-force login burst followed by a large domestic transfer.
        amount=3000 (+25) + logins=8 (+20) = 45 → medium
        """
        score = score_transaction(tx(
            device_risk_score=0,
            is_international=0,
            amount_usd=3000,
            velocity_24h=1,
            failed_logins_24h=8,
            prior_chargebacks=0,
        ))
        assert score == 45
        assert label_risk(score) == "medium"

    def test_account_takeover_with_velocity_burst(self):
        """Login failures AND high velocity: classic ATO with rapid drain.
        device=45 (+15) + amount=600 (+10) + vel=7 (+25) + logins=5 (+20) = 70
        """
        assert score_transaction(tx(
            device_risk_score=45,
            is_international=0,
            amount_usd=600,
            velocity_24h=7,
            failed_logins_24h=5,
            prior_chargebacks=0,
        )) == 70

    def test_repeat_fraudster_large_purchase(self):
        """Serial offender (2+ chargebacks) on a mid-risk device making a large purchase.
        device=45 (+15) + amount=1200 (+25) + chargebacks=3 (+30) = 70
        """
        score = score_transaction(tx(
            device_risk_score=45,
            is_international=0,
            amount_usd=1200,
            velocity_24h=2,
            failed_logins_24h=0,
            prior_chargebacks=3,
        ))
        assert score == 70
        assert label_risk(score) == "high"

    def test_international_card_not_present_fraud(self):
        """High-risk device, cross-border, large amount, moderate velocity.
        device=72 (+30) + intl (+15) + amount=4500 (+25) + vel=4 (+10) + compound (+10) = 90
        """
        assert score_transaction(tx(
            device_risk_score=72,
            is_international=1,
            amount_usd=4500,
            velocity_24h=4,
            failed_logins_24h=0,
            prior_chargebacks=0,
        )) == 90

    def test_compromised_device_international_large_purchase(self):
        """Known-bad device used internationally for a large purchase.
        device=80 (+30) + intl (+15) + amount=1200 (+25) = 70
        """
        score = score_transaction(tx(
            device_risk_score=80,
            is_international=1,
            amount_usd=1200,
            velocity_24h=2,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 70
        assert label_risk(score) == "high"

    def test_worst_case_all_signals_maxed(self):
        """Every signal at its most dangerous value — must hit the 100 cap.
        device (+30) + intl (+15) + amount (+35) + vel (+25) + logins (+20) + cb (+30) + compound (+10) = 165 → 100
        """
        assert score_transaction(tx(
            device_risk_score=90,
            is_international=1,
            amount_usd=8000,
            velocity_24h=8,
            failed_logins_24h=7,
            prior_chargebacks=2,
        )) == 100

    def test_very_large_outlier_amount(self):
        """$50k transaction on a mildly suspicious device.
        device=45 (+15) + amount=50000 (+35) = 50 → medium
        """
        score = score_transaction(tx(
            device_risk_score=45,
            is_international=0,
            amount_usd=50_000,
            velocity_24h=1,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 50
        assert label_risk(score) == "medium"


# ---------------------------------------------------------------------------
# Legitimate transactions — should NOT be labelled high
# ---------------------------------------------------------------------------

class TestLegitimateTransactions:
    def test_everyday_small_domestic_purchase(self):
        """Trusted device, small amount, clean history — should score 0."""
        score = score_transaction(tx(
            device_risk_score=5,
            is_international=0,
            amount_usd=45,
            velocity_24h=1,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 0
        assert label_risk(score) == "low"

    def test_routine_large_purchase_clean_account(self):
        """Single large purchase on a clean account with no red flags.
        amount=1500 (+25) = 25 → low
        """
        score = score_transaction(tx(
            device_risk_score=10,
            is_international=0,
            amount_usd=1500,
            velocity_24h=1,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 25
        assert label_risk(score) == "low"

    def test_international_traveler_single_purchase(self):
        """One cross-border transaction at moderate amount — traveler, not fraud.
        intl (+15) = 15 → low
        """
        score = score_transaction(tx(
            device_risk_score=5,
            is_international=1,
            amount_usd=200,
            velocity_24h=2,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 15
        assert label_risk(score) == "low"

    def test_forgotten_password_small_purchase(self):
        """Two failed logins (mistyped password), then a tiny domestic purchase.
        logins=2 (+10) = 10 → low
        """
        score = score_transaction(tx(
            device_risk_score=0,
            is_international=0,
            amount_usd=30,
            velocity_24h=1,
            failed_logins_24h=2,
            prior_chargebacks=0,
        ))
        assert score == 10
        assert label_risk(score) == "low"

    def test_active_user_moderate_velocity(self):
        """Power user making several small domestic transactions in a day.
        vel=4 (+10) = 10 → low
        """
        score = score_transaction(tx(
            device_risk_score=10,
            is_international=0,
            amount_usd=80,
            velocity_24h=4,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 10
        assert label_risk(score) == "low"

    def test_one_prior_chargeback_small_clean_purchase(self):
        """Single historical chargeback (possibly a legitimate dispute) + low-risk profile.
        chargebacks=1 (+15) = 15 → low, certainly not high
        """
        score = score_transaction(tx(
            device_risk_score=5,
            is_international=0,
            amount_usd=75,
            velocity_24h=1,
            failed_logins_24h=0,
            prior_chargebacks=1,
        ))
        assert score == 15
        assert label_risk(score) == "low"

    def test_international_traveler_high_spend_not_high_risk(self):
        """Tourist spending abroad — international + large amount but no other flags.
        intl (+15) + amount=1200 (+25) = 40 → medium, not high
        """
        score = score_transaction(tx(
            device_risk_score=0,
            is_international=1,
            amount_usd=1200,
            velocity_24h=2,
            failed_logins_24h=0,
            prior_chargebacks=0,
        ))
        assert score == 40
        assert label_risk(score) == "medium"


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

class TestBuildModelFrame:
    def _tx(self, **kw) -> dict:
        base = {
            "transaction_id": "t1",
            "account_id": "a1",
            "amount_usd": 100,
            "failed_logins_24h": 0,
            "velocity_24h": 1,
        }
        base.update(kw)
        return base

    def _acct(self, **kw) -> dict:
        base = {"account_id": "a1"}
        base.update(kw)
        return base

    # --- merge ---

    def test_merge_brings_in_account_columns(self):
        txns, accts = make_frames(
            [self._tx()],
            [self._acct(account_age_days=365)],
        )
        result = build_model_frame(txns, accts)
        assert "account_age_days" in result.columns
        assert result.loc[0, "account_age_days"] == 365

    def test_unmatched_transaction_gets_nan_account_fields(self):
        txns, accts = make_frames(
            [self._tx(account_id="unknown")],
            [self._acct(account_age_days=365)],
        )
        result = build_model_frame(txns, accts)
        assert pd.isna(result.loc[0, "account_age_days"])

    # --- is_large_amount ---

    def test_is_large_amount_below_threshold(self):
        txns, accts = make_frames([self._tx(amount_usd=999)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert result.loc[0, "is_large_amount"] == 0

    def test_is_large_amount_at_threshold(self):
        txns, accts = make_frames([self._tx(amount_usd=1000)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert result.loc[0, "is_large_amount"] == 1

    def test_is_large_amount_above_threshold(self):
        txns, accts = make_frames([self._tx(amount_usd=5000)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert result.loc[0, "is_large_amount"] == 1

    # --- login_pressure ---

    def test_login_pressure_none_at_zero(self):
        txns, accts = make_frames([self._tx(failed_logins_24h=0)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert str(result.loc[0, "login_pressure"]) == "none"

    def test_login_pressure_low_at_one(self):
        txns, accts = make_frames([self._tx(failed_logins_24h=1)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert str(result.loc[0, "login_pressure"]) == "low"

    def test_login_pressure_low_at_two(self):
        txns, accts = make_frames([self._tx(failed_logins_24h=2)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert str(result.loc[0, "login_pressure"]) == "low"

    def test_login_pressure_high_at_three(self):
        txns, accts = make_frames([self._tx(failed_logins_24h=3)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert str(result.loc[0, "login_pressure"]) == "high"

    def test_login_pressure_high_at_five(self):
        txns, accts = make_frames([self._tx(failed_logins_24h=5)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert str(result.loc[0, "login_pressure"]) == "high"

    # --- velocity_spike ---

    def test_velocity_spike_defaults_to_zero_without_account_history(self):
        txns, accts = make_frames([self._tx(velocity_24h=50)], [self._acct()])
        result = build_model_frame(txns, accts)
        assert result.loc[0, "velocity_spike"] == 0

    def test_velocity_spike_flagged_when_exceeds_3x_average(self):
        # velocity=10, avg=2 → 10 > 6 → spike
        txns, accts = make_frames(
            [self._tx(velocity_24h=10)],
            [self._acct(avg_daily_velocity=2)],
        )
        result = build_model_frame(txns, accts)
        assert result.loc[0, "velocity_spike"] == 1

    def test_velocity_spike_not_flagged_within_normal_range(self):
        # velocity=5, avg=4 → 5 <= 12 → no spike
        txns, accts = make_frames(
            [self._tx(velocity_24h=5)],
            [self._acct(avg_daily_velocity=4)],
        )
        result = build_model_frame(txns, accts)
        assert result.loc[0, "velocity_spike"] == 0

    def test_velocity_spike_at_exact_boundary_is_not_flagged(self):
        # velocity=6, avg=2 → 6 == 6, condition is strictly >, so no spike
        txns, accts = make_frames(
            [self._tx(velocity_24h=6)],
            [self._acct(avg_daily_velocity=2)],
        )
        result = build_model_frame(txns, accts)
        assert result.loc[0, "velocity_spike"] == 0

    def test_velocity_spike_just_above_boundary_is_flagged(self):
        # velocity=7, avg=2 → 7 > 6 → spike
        txns, accts = make_frames(
            [self._tx(velocity_24h=7)],
            [self._acct(avg_daily_velocity=2)],
        )
        result = build_model_frame(txns, accts)
        assert result.loc[0, "velocity_spike"] == 1

    # --- multi-row correctness ---

    def test_multiple_rows_scored_independently(self):
        txns, accts = make_frames(
            [
                self._tx(transaction_id="t1", amount_usd=500),
                {"transaction_id": "t2", "account_id": "a2",
                 "amount_usd": 1500, "failed_logins_24h": 0, "velocity_24h": 1},
            ],
            [self._acct(), {"account_id": "a2"}],
        )
        result = build_model_frame(txns, accts)
        assert result.loc[0, "is_large_amount"] == 0
        assert result.loc[1, "is_large_amount"] == 1
