"""End-to-end pipeline test in offline mode (in-memory store, no API key)."""
from pathlib import Path

from matcher.config import Settings
from matcher.engine import build_engine

DATA = Path(__file__).parent.parent / "data" / "sample"


def make_engine():
    return build_engine(Settings(openai_api_key="", vector_store="memory", redis_url=""))


def test_pipeline_ranks_the_obvious_candidate_first():
    engine = make_engine()
    for p in sorted((DATA / "resumes").glob("*.txt")):
        engine.add_resume(p.name, p.read_bytes(), resume_id=p.name)
    job_id, job = engine.add_job("data_analyst.txt", (DATA / "jobs" / "data_analyst.txt").read_bytes())
    assert "power bi" in job.required_skills

    results = engine.match(job_id, top_n=5)
    assert results[0].resume_id == "cv04.txt"
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert results[0].score >= results[-1].score
    assert results[0].explanation and results[0].explanation.strengths


def test_reupload_does_not_duplicate_and_pii_is_not_stored():
    engine = make_engine()
    data = (DATA / "resumes" / "cv01.txt").read_bytes()
    id1, _ = engine.add_resume("cv01.txt", data)
    id2, _ = engine.add_resume("cv01.txt", data)
    assert id1 == id2 and len(engine.store.list_ids("resumes")) == 1
    payload = engine.store.get("resumes", id1)[1]
    assert "example.com" not in str(payload) and "Karim" not in str(payload)
