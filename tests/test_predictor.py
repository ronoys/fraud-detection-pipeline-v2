import pytest
from unittest.mock import MagicMock, patch
from conftest import SAMPLE_TRANSACTION, _mock_model, _mock_scaler


def setup_predictor(fraud_prob=0.8, scaled=0.42, no_challenger=False):
    import predictor
    predictor._champion = _mock_model(fraud_prob)
    predictor._challenger = None if no_challenger else _mock_model(fraud_prob)
    predictor._scaler = _mock_scaler(scaled)
    return predictor


class TestPredict:
    def test_returns_four_tuple(self):
        p = setup_predictor()
        result = p.predict(SAMPLE_TRANSACTION)
        assert len(result) == 4
        is_fraud, confidence, scaled_amount, model_name = result
        assert isinstance(is_fraud, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(scaled_amount, float)
        assert model_name in ("xgboost", "random_forest")

    def test_high_confidence_is_fraud(self):
        p = setup_predictor(fraud_prob=0.95)
        is_fraud, confidence, _, _ = p.predict(SAMPLE_TRANSACTION)
        assert is_fraud is True
        assert confidence == pytest.approx(0.95)

    def test_low_confidence_is_legit(self):
        p = setup_predictor(fraud_prob=0.1)
        is_fraud, _, _, _ = p.predict(SAMPLE_TRANSACTION)
        assert is_fraud is False

    def test_exactly_at_threshold_is_fraud(self):
        p = setup_predictor(fraud_prob=0.5)
        is_fraud, _, _, _ = p.predict(SAMPLE_TRANSACTION)
        assert is_fraud is True

    def test_amount_is_scaled(self):
        p = setup_predictor(scaled=1.23)
        _, _, scaled_amount, _ = p.predict(SAMPLE_TRANSACTION)
        assert scaled_amount == pytest.approx(1.23)

    def test_scaled_amount_passed_to_model(self):
        p = setup_predictor(scaled=0.99)
        p.predict(SAMPLE_TRANSACTION)
        call_args = p._champion.predict_proba.call_args
        X = call_args[0][0]
        assert float(X["Amount"].iloc[0]) == pytest.approx(0.99)

    def test_prediction_logged(self):
        p = setup_predictor()
        p.prediction_log.clear()
        p.predict(SAMPLE_TRANSACTION)
        assert len(p.prediction_log) == 1
        entry = p.prediction_log[0]
        assert "confidence" in entry
        assert "model" in entry
        assert "Amount" in entry

    def test_no_challenger_always_uses_champion(self):
        p = setup_predictor(no_challenger=True)
        for _ in range(20):
            _, _, _, model_name = p.predict(SAMPLE_TRANSACTION)
            assert model_name == "xgboost"

    def test_raises_if_not_loaded(self):
        import predictor
        predictor._champion = None
        predictor._scaler = None
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.predict(SAMPLE_TRANSACTION)
