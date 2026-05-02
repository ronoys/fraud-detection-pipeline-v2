import pytest
from pydantic import ValidationError
from schemas import DriftResponse, PredictionResponse, TransactionRequest


def _base_tx(**overrides):
    return {
        "Time": 1000.0,
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        **overrides,
    }


class TestTransactionRequest:
    def test_valid(self):
        tx = TransactionRequest(**_base_tx())
        assert tx.Amount == 100.0
        assert tx.Time == 1000.0

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_base_tx(Amount=-1.0))

    def test_zero_amount_accepted(self):
        tx = TransactionRequest(**_base_tx(Amount=0.0))
        assert tx.Amount == 0.0

    def test_missing_v_feature_rejected(self):
        data = _base_tx()
        del data["V14"]
        with pytest.raises(ValidationError):
            TransactionRequest(**data)

    def test_missing_amount_rejected(self):
        data = _base_tx()
        del data["Amount"]
        with pytest.raises(ValidationError):
            TransactionRequest(**data)


class TestPredictionResponse:
    def test_model_used_field_present(self):
        resp = PredictionResponse(
            fraud=True,
            confidence=0.9,
            transaction_id="abc-123",
            timestamp="2024-01-01T00:00:00+00:00",
            model_used="xgboost",
        )
        assert resp.model_used == "xgboost"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                fraud=False,
                confidence=1.5,
                transaction_id="x",
                timestamp="2024-01-01T00:00:00+00:00",
            )


class TestDriftResponse:
    def test_valid(self):
        d = DriftResponse(
            drift_detected=True,
            drifted_features=3,
            total_features=29,
            share_drifted=0.103,
            feature_drift_scores={"Amount": 0.12, "V1": 0.08},
            predictions_analyzed=200,
        )
        assert d.drift_detected is True
        assert d.share_drifted == 0.103
