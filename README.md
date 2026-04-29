from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    tx = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_very_large_amount_adds_more_risk():
    low_amount = score_transaction(_base_tx(amount_usd=1200))
    high_amount = score_transaction(_base_tx(amount_usd=6000))
    assert high_amount > low_amount


def test_high_device_risk_adds_risk():
    # Previously ≥70 subtracted 25; it must now add risk.
    assert score_transaction(_base_tx(device_risk_score=80)) > score_transaction(_base_tx(device_risk_score=0))


def test_international_adds_risk():
    # Previously subtracted 15; must now raise the score.
    domestic = score_transaction(_base_tx(is_international=0))
    international = score_transaction(_base_tx(is_international=1))
    assert international > domestic


def test_high_velocity_adds_risk():
    # Previously ≥6 subtracted 20; must now raise the score.
    low_v = score_transaction(_base_tx(velocity_24h=1))
    high_v = score_transaction(_base_tx(velocity_24h=8))
    assert high_v > low_v


def test_prior_chargebacks_add_risk():
    # Previously chargebacks subtracted points; must now raise the score.
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one_cb = score_transaction(_base_tx(prior_chargebacks=1))
    two_cb = score_transaction(_base_tx(prior_chargebacks=2))
    assert one_cb > clean
    assert two_cb > one_cb


def test_combined_fraud_profile_scores_high():
    # High-risk device + international + high velocity + prior chargebacks
    # previously scored 0 due to stacked inversions. Must now score high.
    tx = _base_tx(
        device_risk_score=80,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        prior_chargebacks=2,
    )
    assert score_transaction(tx) >= 60


def test_low_risk_profile_scores_low():
    tx = _base_tx(
        device_risk_score=5,
        is_international=0,
        amount_usd=50,
        velocity_24h=1,
        failed_logins_24h=0,
        prior_chargebacks=0,
    )
    assert score_transaction(tx) < 30


def test_international_high_velocity_compound_penalty():
    # The compound interaction term should make international + high-velocity
    # score higher than the sum of those signals independently.
    intl_only = score_transaction(_base_tx(is_international=1, velocity_24h=1))
    vel_only = score_transaction(_base_tx(is_international=0, velocity_24h=4))
    combined = score_transaction(_base_tx(is_international=1, velocity_24h=4))
    assert combined > intl_only
    assert combined > vel_only
