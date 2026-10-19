import os
import sys
from pathlib import Path

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HEADERS = {"X-API-Key": os.getenv("API_KEY", "")}
DATA = Path(__file__).resolve().parent.parent / "data" / "sample"


def main() -> None:
    health = requests.get(f"{BASE}/health", timeout=10).json()
    print("health:", health)
    assert health["status"] == "ok"

    files = [("files", (p.name, p.read_bytes())) for p in sorted((DATA / "resumes").glob("*.txt"))]
    r = requests.post(f"{BASE}/resumes", files=files, headers=HEADERS, timeout=600)
    r.raise_for_status()
    added = r.json()["added"]
    assert len(added) == len(files), r.json()["errors"]

    job_path = DATA / "jobs" / "data_analyst.txt"
    r = requests.post(f"{BASE}/jobs", files={"file": (job_path.name, job_path.read_bytes())},
                      headers=HEADERS, timeout=120)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    r = requests.get(f"{BASE}/jobs/{job_id}/matches", params={"top_n": 3}, headers=HEADERS, timeout=300)
    r.raise_for_status()
    top = r.json()
    print("top 3:", [(m["filename"], m["score"]) for m in top])
    assert len(top) == 3 and top[0]["filename"] == "cv04.txt", "unexpected ranking"

    # auth must be enforced when an API key is configured
    if HEADERS["X-API-Key"]:
        assert requests.get(f"{BASE}/jobs/{job_id}/matches", timeout=10).status_code == 401

    for m in added:                          # clean up test data
        requests.delete(f"{BASE}/resumes/{m['resume_id']}", headers=HEADERS, timeout=10)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("SMOKE TEST FAILED:", e)
        sys.exit(1)
