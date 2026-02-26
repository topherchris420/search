from __future__ import annotations


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["documents_indexed"] >= 1


def test_filters_require_auth(client):
    response = client.get("/api/filters")
    assert response.status_code == 401


def test_search_success_and_cache(client, auth_headers):
    payload = {
        "query": "rf anomaly mission corridor",
        "filters": {
            "category": "all",
            "source": "all",
            "security_tier": "all",
            "ontology_type": "all",
        },
        "page": 1,
        "page_size": 5,
    }

    first = client.post("/api/search", json=payload, headers=auth_headers)
    assert first.status_code == 200
    first_body = first.get_json()
    assert isinstance(first_body["results"], list)
    assert first_body["cache_hit"] is False

    second = client.post("/api/search", json=payload, headers=auth_headers)
    assert second.status_code == 200
    second_body = second.get_json()
    assert second_body["cache_hit"] is True


def test_filters_and_pagination(client, auth_headers):
    payload = {
        "query": "",
        "filters": {
            "category": "policy",
            "source": "all",
            "security_tier": "all",
            "ontology_type": "all",
        },
        "page": 1,
        "page_size": 2,
    }

    response = client.post("/api/search", json=payload, headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["page_size"] == 2
    assert body["total"] >= 1
    assert all(result["category"] == "policy" for result in body["results"])


def test_invalid_payload(client, auth_headers):
    response = client.post("/api/search", json={"query": "x", "page": "bad"}, headers=auth_headers)
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body


def test_reindex(client, auth_headers):
    response = client.post("/api/reindex", headers=auth_headers)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] >= 1
