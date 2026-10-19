from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from matcher.config import get_settings

DATA = Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("API_KEY", "test-key")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_auth_required(client):
    r = client.post("/jobs", files={"file": ("job.txt", b"x" * 100)})
    assert r.status_code == 401


def test_full_flow(client):
    h = {"X-API-Key": "test-key"}
    files = [("files", (p.name, p.read_bytes())) for p in sorted((DATA / "resumes").glob("*.txt"))]
    files.append(("files", ("photo.png", b"not a resume")))
    r = client.post("/resumes", files=files, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert len(body["added"]) == 12 and body["errors"][0]["filename"] == "photo.png"

    job = client.post("/jobs", files={"file": ("backend.txt", (DATA / "jobs" / "backend_python.txt").read_bytes())},
                      headers=h).json()
    matches = client.get(f"/jobs/{job['job_id']}/matches", params={"top_n": 3}, headers=h).json()
    assert len(matches) == 3 and matches[0]["filename"] == "cv06.txt"

    assert client.get("/jobs/unknown/matches", headers=h).status_code == 404
    assert client.get("/health").json()["resumes"] == 12
