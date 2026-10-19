"""Compare the matching pipeline with simpler baselines on labeled data.

    python -m eval.evaluate                 # uses OpenAI if OPENAI_API_KEY is set, else offline mode
    python -m eval.evaluate --offline       # force offline (rule-based + hashing embeddings)

Systems compared
    keyword_tfidf  : TF-IDF cosine between raw texts (classic ATS-style baseline)
    semantic_only  : embedding similarity only
    full_pipeline  : LLM extraction + embeddings + explainable skill/experience scoring

Writes eval/results.md. Only numbers produced with real OpenAI models are meaningful;
offline mode is for checking the code runs.
"""
import json
import sys
from pathlib import Path
from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval.metrics import ndcg_at_k, precision_at_k
from matcher.config import Settings, get_settings
from matcher.engine import build_engine

DATA = Path("data/sample")
K = 3


def load(folder: str) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((DATA / folder).glob("*.txt"))}


def main(offline: bool = False) -> None:
    labels = {k: v for k, v in json.loads(Path("eval/labels.json").read_text()).items() if not k.startswith("_")}
    jobs, resumes = load("jobs"), load("resumes")

    base = get_settings()
    settings = Settings(**{**base.model_dump(), "vector_store": "memory",
                           **({"openai_api_key": ""} if offline else {})})
    mode = "OpenAI" if settings.openai_api_key else "offline (rule-based + hashing, NOT meaningful)"
    engine = build_engine(settings)

    for name, text in resumes.items():
        engine.add_resume(name, text.encode(), resume_id=name)
    job_ids = {name: engine.add_job(name, text.encode(), job_id=name)[0] for name, text in jobs.items()}

    tfidf = TfidfVectorizer(stop_words="english").fit(list(jobs.values()) + list(resumes.values()))
    names = list(resumes)
    R = tfidf.transform([resumes[n] for n in names])

    rows = {s: {"ndcg": [], "p": []} for s in ["keyword_tfidf", "semantic_only", "full_pipeline"]}
    for job_name, job_labels in labels.items():
        sims = cosine_similarity(tfidf.transform([jobs[job_name]]), R)[0]
        rankings = {"keyword_tfidf": [n for _, n in sorted(zip(sims, names), reverse=True)]}

        results = engine.match(job_ids[job_name], top_n=len(names), explain=False)
        rankings["semantic_only"] = [r.resume_id for r in sorted(results, key=lambda r: -r.breakdown.semantic)]
        rankings["full_pipeline"] = [r.resume_id for r in results]

        for system, ranked in rankings.items():
            rows[system]["ndcg"].append(ndcg_at_k(ranked, job_labels, K))
            rows[system]["p"].append(precision_at_k(ranked, job_labels, K))
        print(f"{job_name:22s} top-3 full pipeline: {rankings['full_pipeline'][:3]}")

    lines = ["# Evaluation results\n", f"Mode: {mode}  \nJobs: {len(labels)} · Resumes: {len(names)}\n",
             f"| System | NDCG@{K} | Precision@{K} |", "|---|---|---|"]
    for system, m in rows.items():
        lines.append(f"| {system} | {mean(m['ndcg']):.3f} | {mean(m['p']):.3f} |")
    report = "\n".join(lines) + "\n"
    Path("eval/results.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
