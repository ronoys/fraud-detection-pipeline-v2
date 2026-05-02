import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

SAMPLE_TRANSACTION = {
    "Time": 1000.0,
    "Amount": 149.62,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}


def _mock_model(fraud_prob: float = 0.8) -> MagicMock:
    m = MagicMock()
    m.predict_proba.return_value = [[1 - fraud_prob, fraud_prob]]
    return m


def _mock_scaler(scaled: float = 0.42) -> MagicMock:
    m = MagicMock()
    m.transform.return_value = [[scaled]]
    return m


def _make_client(fraud_prob: float):
    import main as main_module
    import predictor as predictor_module

    # main.py imports load_model via "from predictor import load_model", so we must
    # patch the reference in the main module (not in predictor) to intercept the call.
    def fake_load_model():
        predictor_module._champion = _mock_model(fraud_prob=fraud_prob)
        predictor_module._challenger = _mock_model(fraud_prob=fraud_prob)
        predictor_module._scaler = _mock_scaler()

    main_module._alerts.clear()
    predictor_module.prediction_log.clear()

    with patch("main.load_model", side_effect=fake_load_model):
        from main import app
        with TestClient(app) as c:
            predictor_module.prediction_log.clear()  # clear any warmup entries
            yield c

    predictor_module._champion = None
    predictor_module._challenger = None
    predictor_module._scaler = None


@pytest.fixture(autouse=True)
def clear_state():
    import main as main_module
    import predictor as predictor_module
    predictor_module.prediction_log.clear()
    main_module._alerts.clear()
    yield
    predictor_module.prediction_log.clear()
    main_module._alerts.clear()


@pytest.fixture
def client():
    yield from _make_client(fraud_prob=0.8)


@pytest.fixture
def client_fraud_below_threshold():
    yield from _make_client(fraud_prob=0.3)
