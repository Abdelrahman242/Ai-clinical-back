import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Isolate the preflight run from production credentials and background workers.
os.environ.pop("SUPABASE_DB_URL", None)
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.gettempdir()) / "clinical_preflight.db")
os.environ["AUTO_INGEST_ENABLED"] = "false"
os.environ["JWT_SECRET_KEY"] = "preflight-test-secret"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models
from app.database import Base, get_db
from app.main import app
from app.routers import conversations as conversations_router
from app.routers import documents as documents_router
from app.routers import evaluations as evaluations_router
from app.routers import retrieve as retrieve_router

TEST_DB = Path(tempfile.gettempdir()) / "clinical_preflight.db"
if TEST_DB.exists():
    TEST_DB.unlink()

engine = create_engine(
    "sqlite:///" + str(TEST_DB), connect_args={"check_same_thread": False}
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def fake_pipeline(**kwargs):
    citation = SimpleNamespace(document="WHO Hypertension", section="Overview", page=1, chunk_id="c1")
    return SimpleNamespace(
        answer="Grounded test answer",
        citations=[citation],
        confidence="High",
        refused=False,
        risk_flag="allowed",
        max_retrieval_score=0.9,
    )


app.dependency_overrides[get_db] = override_get_db
conversations_router.run_pipeline = fake_pipeline
evaluations_router.run_pipeline = fake_pipeline
retrieve_router.similarity_search_with_score = lambda project_id, query, k: []
documents_router.run_ingest_job = lambda job_id, document_id, reset=False: None

with TestClient(app) as client:
    first = client.post(
        "/api/v1/auth/register",
        json={"username": "first", "email": "first@example.com", "password": "password123"},
    )
    assert first.status_code == 200, first.text
    second = client.post(
        "/api/v1/auth/register",
        json={"username": "second", "email": "second@example.com", "password": "password123"},
    )
    assert second.status_code == 200, second.text

    login = client.post(
        "/api/v1/auth/login",
        data={"username": "first", "password": "password123"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200 and me.json()["username"] == "first", me.text

    update = client.patch(
        "/api/v1/auth/me", headers=headers, json={"name": "first-renamed"}
    )
    assert update.status_code == 200, update.text
    assert update.json()["username"] == "first-renamed", update.text
    rotated_token = update.json()["access_token"]
    rotated_headers = {"Authorization": f"Bearer {rotated_token}"}
    assert client.get("/api/v1/auth/me", headers=rotated_headers).status_code == 200

    project = client.post(
        "/api/v1/projects",
        headers=rotated_headers,
        json={"name": "Test Project", "clinical_topic": "Hypertension"},
    )
    assert project.status_code == 200, project.text
    project_id = project.json()["id"]

    assert client.get("/api/v1/projects", headers=rotated_headers).status_code == 200
    assert client.get(
        f"/api/v1/projects/{project_id}", headers=rotated_headers
    ).status_code == 200

    document = client.post(
        f"/api/v1/projects/{project_id}/documents",
        headers=rotated_headers,
        json={"title": "Test guideline", "source_ref": "missing.pdf"},
    )
    assert document.status_code == 200, document.text
    document_id = document.json()["id"]
    assert client.get(
        f"/api/v1/projects/{project_id}/documents", headers=rotated_headers
    ).status_code == 200
    assert client.get(
        f"/api/v1/documents/{document_id}/status", headers=rotated_headers
    ).status_code == 200

    reindex = client.post(
        f"/api/v1/projects/{project_id}/reindex", headers=rotated_headers
    )
    assert reindex.status_code == 200, reindex.text
    assert reindex.json()["project_id"] == project_id
    assert len(reindex.json()["queued_jobs"]) == 1

    retrieve = client.post(
        f"/api/v1/projects/{project_id}/retrieve",
        headers=rotated_headers,
        json={"query": "blood pressure", "top_k": 3},
    )
    assert retrieve.status_code == 200, retrieve.text

    evaluation = client.post(
        f"/api/v1/projects/{project_id}/evaluations",
        headers=rotated_headers,
        json={"cases": [{"question": "blood pressure", "expected_keywords": ["WHO"]}]},
    )
    assert evaluation.status_code == 200, evaluation.text

    conversation = client.post(
        f"/api/v1/projects/{project_id}/conversations",
        headers=rotated_headers,
        json={"title": "Private chat"},
    )
    assert conversation.status_code == 200, conversation.text
    conversation_id = conversation.json()["id"]

    second_login = client.post(
        "/api/v1/auth/login",
        data={"username": "second", "password": "password123"},
    )
    second_headers = {"Authorization": f"Bearer {second_login.json()['access_token']}"}
    assert client.get(
        f"/api/v1/conversations/{conversation_id}/messages", headers=second_headers
    ).status_code == 404

    message = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        headers=rotated_headers,
        json={"query": "What is blood pressure?", "top_k": 3},
    )
    assert message.status_code == 200, message.text
    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages", headers=rotated_headers
    )
    assert messages.status_code == 200, messages.text
    assert [item["role"] for item in messages.json()] == ["user", "assistant"]

    ingest = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers=rotated_headers,
        json={"reset": False},
    )
    assert ingest.status_code == 200, ingest.text
    assert client.get(
        f"/api/v1/jobs/{ingest.json()['id']}", headers=rotated_headers
    ).status_code == 200

    logout = client.post("/api/v1/auth/logout", headers=rotated_headers)
    assert logout.status_code == 200, logout.text
    assert client.get("/api/v1/auth/me", headers=rotated_headers).status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200

app.dependency_overrides.clear()
if TEST_DB.exists():
    TEST_DB.unlink()
print("full_api_preflight=passed")
