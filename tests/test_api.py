from conftest import SAMPLE_TRANSACTION


class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["champion"] == "xgboost"

    def test_logs_prediction_count(self, client):
        client.post("/predict", json=SAMPLE_TRANSACTION)
        r = client.get("/health")
        assert r.json()["predictions_logged"] == 1


class TestPredict:
    def test_valid_transaction_returns_200(self, client):
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert r.status_code == 200

    def test_response_shape(self, client):
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        body = r.json()
        assert "fraud" in body
        assert "confidence" in body
        assert "transaction_id" in body
        assert "timestamp" in body
        assert "pipeline_steps" in body
        assert "model_used" in body

    def test_model_used_is_valid(self, client):
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert r.json()["model_used"] in ("xgboost", "random_forest")

    def test_high_confidence_flagged_as_fraud(self, client):
        # fixture model returns fraud_prob=0.8 which is ≥ 0.5 threshold
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert r.json()["fraud"] is True

    def test_low_confidence_flagged_as_legit(self, client_fraud_below_threshold):
        r = client_fraud_below_threshold.post("/predict", json=SAMPLE_TRANSACTION)
        assert r.json()["fraud"] is False

    def test_pipeline_steps_present(self, client):
        r = client.post("/predict", json=SAMPLE_TRANSACTION)
        steps = r.json()["pipeline_steps"]
        assert len(steps) == 4
        assert all("name" in s and "detail" in s for s in steps)

    def test_missing_field_returns_422(self, client):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V14"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_amount_returns_422(self, client):
        bad = {**SAMPLE_TRANSACTION, "Amount": -50.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_fraud_creates_alert(self, client):
        client.post("/predict", json=SAMPLE_TRANSACTION)
        alerts = client.get("/alerts").json()
        assert len(alerts) == 1
        assert alerts[0]["confidence"] == 0.8

    def test_legit_does_not_create_alert(self, client_fraud_below_threshold):
        client_fraud_below_threshold.post("/predict", json=SAMPLE_TRANSACTION)
        alerts = client_fraud_below_threshold.get("/alerts").json()
        assert len(alerts) == 0


class TestModelStats:
    def test_returns_200(self, client):
        r = client.get("/model-stats")
        assert r.status_code == 200

    def test_shape(self, client):
        r = client.get("/model-stats")
        body = r.json()
        assert "champion" in body
        assert "challenger" in body
        assert "total_predictions" in body

    def test_counts_increment(self, client):
        client.post("/predict", json=SAMPLE_TRANSACTION)
        client.post("/predict", json=SAMPLE_TRANSACTION)
        r = client.get("/model-stats")
        assert r.json()["total_predictions"] == 2


class TestAlerts:
    def test_returns_list(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_capped_at_25(self, client):
        for _ in range(30):
            client.post("/predict", json=SAMPLE_TRANSACTION)
        r = client.get("/alerts")
        assert len(r.json()) <= 25


class TestDrift:
    def test_requires_50_predictions(self, client):
        r = client.get("/drift")
        assert r.status_code == 400
        assert "50" in r.json()["detail"]
