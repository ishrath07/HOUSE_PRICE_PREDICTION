from fastapi.testclient import TestClient

from house_price import api, pipeline


def test_health_and_prediction_endpoint(ames):
    X, y = ames
    fitted_model = pipeline.build_model().fit(X, y)
    app = api.create_app(model=fitted_model)
    record = X.iloc[0].astype(object).where(X.iloc[0].notna(), None).to_dict()

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=record)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert prediction.status_code == 200
    assert prediction.json()["predicted_sale_price"] > 0


def test_prediction_endpoint_returns_422_for_missing_required_feature(ames):
    X, y = ames
    fitted_model = pipeline.build_model().fit(X, y)
    row = X.iloc[0].astype(object)
    record = row.drop(labels=["GrLivArea"]).where(row.notna(), None).to_dict()

    with TestClient(api.create_app(model=fitted_model)) as client:
        response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert "GrLivArea" in response.json()["detail"]
