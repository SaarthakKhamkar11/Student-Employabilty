from fastapi.testclient import TestClient

from app.agents import build_agents
from app.core.config import get_settings
from app.db.store import InMemoryAgentRunStore
from app.main import app, get_supervisor
from app.schemas.agent_io import SupervisorResponse

client = TestClient(app)


def test_health_reports_no_llm_calls_and_leaks_no_secrets():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["llm_calls_enabled"] is get_settings().llm_configured
    assert set(body["dependencies"]) == {"postgres", "redis"}
    s = get_settings()
    for secret in (s.database_url, s.jwt_secret, s.gemini_api_key):
        if secret and secret.get_secret_value():
            assert secret.get_secret_value() not in resp.text


def test_agents_endpoint_lists_the_supervisor_and_five_shells():
    agents = {a["name"]: a for a in client.get("/agents").json()}
    assert len(agents) == 6 and agents["supervisor"]["implemented"] is True
    assert all(not a["implemented"] for name, a in agents.items() if name != "supervisor")


def test_post_request_runs_the_supervisor_and_the_run_can_be_read_back():
    store = InMemoryAgentRunStore()
    app.dependency_overrides[get_supervisor] = lambda: build_agents(store=store)["supervisor"]
    try:
        resp = client.post("/requests", json={"actor": "company", "request_kind": "jd_upload",
                                              "jd_text": "Data Analyst. Required: Python, SQL."})
        assert resp.status_code == 200
        body = SupervisorResponse.model_validate(resp.json())
        assert body.task_type == "ingest_job" and body.persisted
        stored = client.get(f"/requests/{body.request_id}")
        assert stored.status_code == 200 and stored.json()["status"] == body.status
        assert client.get("/requests/00000000-0000-0000-0000-000000000000").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_malformed_request_is_rejected():
    assert client.post("/requests", json={"actor": "student", "request_kind": "resume_upload"}).status_code == 422
