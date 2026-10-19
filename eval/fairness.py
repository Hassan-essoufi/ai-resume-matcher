"""Name-swap fairness test (counterfactual check).

Same resume, only the candidate's name changes (different genders and origins).
A fair system gives the same score every time. We run it with PII scrubbing
ON and OFF to show the effect of removing names before scoring.

    python -m eval.fairness [--offline]
"""
import sys
from pathlib import Path
from statistics import mean

from matcher.config import Settings, get_settings
from matcher.engine import build_engine

NAMES = ["Mohamed Alaoui", "Fatima Zahra Bennani", "John Smith", "Emily Chen",
         "Aminata Diallo", "Pierre Dubois", "Kenji Tanaka", "Maria Garcia"]
JOB = "ml_engineer.txt"
DATA = Path("data/sample")


def run(scrub: bool, offline: bool) -> float:
    base = get_settings()
    s = Settings(**{**base.model_dump(), "vector_store": "memory", "scrub_pii": scrub,
                    **({"openai_api_key": ""} if offline else {})})
    engine = build_engine(s)
    job_id, _ = engine.add_job(JOB, (DATA / "jobs" / JOB).read_bytes(), job_id="job")

    spreads = []
    for path in sorted((DATA / "resumes").glob("*.txt"))[:6]:
        body = path.read_text().split("\n", 1)[1]            # drop the original name line
        scores = []
        for i, name in enumerate(NAMES):
            rid = f"{path.stem}-{i}"
            engine.add_resume(path.name, f"{name}\n{body}".encode(), resume_id=rid)
            res = {r.resume_id: r.score for r in engine.match(job_id, top_n=200, explain=False)}
            scores.append(res[rid])
            engine.store.delete("resumes", rid)
        spreads.append(max(scores) - min(scores))
    return max(spreads), mean(spreads)


def main(offline: bool = False) -> None:
    for scrub in (True, False):
        worst, avg = run(scrub, offline)
        print(f"PII scrubbing {'ON ' if scrub else 'OFF'}: score change when only the name changes "
              f"-> max {worst:.2f} pts, mean {avg:.2f} pts (0 = perfectly name-blind)")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
