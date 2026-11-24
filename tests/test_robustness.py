from robustness import evaluate_world


def test_no_signal_oracle_is_chance_like():
    rows = evaluate_world(seed=0, scenario="no_signal", n_days=600)
    assert len(rows) == 2
    for row in rows:
        assert abs(row["oracle_auc"] - 0.5) < 1e-12


def test_nonlinear_world_has_informative_oracle():
    rows = evaluate_world(seed=1, scenario="nonlinear", n_days=800, nonlinear_strength=0.45)
    assert rows[0]["oracle_auc"] > 0.6
