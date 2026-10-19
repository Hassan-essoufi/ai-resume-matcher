import hashlib
import logging
import numpy as np

from matcher import pii
from matcher.cache import make_cache
from matcher.config import get_settings
from matcher.embeddings import HashingEmbedder, OpenAIEmbedder, profile_text
from matcher.extraction import OpenAIExtractor, RuleBasedExtractor
from matcher.ingestion import extract_text
from matcher.llm import OpenAIClient
from matcher.schemas import CandidateProfile, Explanation, JobProfile, MatchResult, ScoreBreakdown
from matcher.scoring import score
from matcher.vector_store import InMemoryStore, QdrantStore

log = logging.getLogger(__name__)

JOBS, RESUMES = "jobs", "resumes"


def template_explanation(job: JobProfile, b: ScoreBreakdown) -> Explanation:
    """Deterministic explanation used when no LLM is configured."""
    strengths = []
    if b.matched_required:
        strengths.append(f"Has {len(b.matched_required)}/{len(job.required_skills)} required skills: "
                         + ", ".join(b.matched_required[:5]))
    if b.experience_fit >= 1 and job.min_years_experience:
        strengths.append(f"Meets the {job.min_years_experience:g}+ years of experience asked")
    if b.matched_nice:
        strengths.append("Bonus skills: " + ", ".join(b.matched_nice[:3]))
    gaps = []
    if b.missing_required:
        gaps.append("Missing required: " + ", ".join(b.missing_required[:5]))
    if b.experience_fit < 1:
        gaps.append("Less experience than required")
    verdict = "Strong match" if b.total >= 75 else "Possible match" if b.total >= 55 else "Weak match"
    return Explanation(strengths=strengths or ["Some overlap with the role"], gaps=gaps,
                       summary=f"{verdict} ({b.total:.0f}/100).")


class MatchingEngine:
    def __init__(self, settings, extractor, embedder, store, explainer=None):
        self.settings = settings
        self.extractor = extractor
        self.embedder = embedder
        self.store = store
        self.explainer = explainer     # OpenAIClient or None (template fallback)

    #  ingestion
    def _prepare(self, filename: str, data: bytes) -> str:
        text = extract_text(filename, data)
        return pii.scrub(text) if self.settings.scrub_pii else text

    def add_job(self, filename: str, data: bytes, job_id: str | None = None) -> tuple[str, JobProfile]:
        text = self._prepare(filename, data)
        profile = self.extractor.job(text)
        vec = self.embedder.embed([profile_text(profile.summary, [profile.title],
                                                profile.required_skills + profile.nice_to_have_skills, text)])[0]
        job_id = job_id or "job_" + hashlib.sha256(data).hexdigest()[:12]   # same file -> same id
        self.store.upsert(JOBS, job_id, vec, {"filename": filename, "profile": profile.model_dump()})
        log.info("job added id=%s required=%d", job_id, len(profile.required_skills))
        return job_id, profile

    def add_resume(self, filename: str, data: bytes, resume_id: str | None = None) -> tuple[str, CandidateProfile]:
        text = self._prepare(filename, data)
        profile = self.extractor.resume(text)
        vec = self.embedder.embed([profile_text(profile.summary, profile.job_titles, profile.skills, text)])[0]
        resume_id = resume_id or "cv_" + hashlib.sha256(data).hexdigest()[:12]   # re-uploads do not duplicate
        self.store.upsert(RESUMES, resume_id, vec, {"filename": filename, "profile": profile.model_dump()})
        log.info("resume added id=%s skills=%d", resume_id, len(profile.skills))
        return resume_id, profile

    # matching
    def get_job(self, job_id: str) -> JobProfile | None:
        rec = self.store.get(JOBS, job_id)
        return JobProfile(**rec[1]["profile"]) if rec else None

    def match(self, job_id: str, top_n: int = 10, explain: bool = True) -> list[MatchResult]:
        rec = self.store.get(JOBS, job_id)
        if rec is None:
            raise KeyError(job_id)
        job_vec, payload = rec
        job = JobProfile(**payload["profile"])

        # 1) fast semantic shortlist, 2) precise explainable scoring on the shortlist
        hits = self.store.search(RESUMES, job_vec, self.settings.retrieve_top_k)
        scored = []
        for resume_id, _, vec, p in hits:
            cand = CandidateProfile(**p["profile"])
            cosine = float(np.dot(job_vec, vec))
            b = score(job, cand, cosine, self.settings, embedder=self.embedder)
            scored.append((resume_id, p["filename"], cand, b))
        scored.sort(key=lambda x: x[3].total, reverse=True)

        results = []
        for rank, (resume_id, filename, cand, b) in enumerate(scored[:top_n], start=1):
            expl = None
            if explain:
                try:
                    expl = self.explainer.explain(job, cand, b) if self.explainer else template_explanation(job, b)
                except Exception:   # an explanation must never break the ranking
                    log.exception("explanation failed for %s", resume_id)
                    expl = template_explanation(job, b)
            results.append(MatchResult(resume_id=resume_id, filename=filename, rank=rank, score=b.total,
                                       breakdown=b, profile=cand, explanation=expl))
        return results


def build_engine(settings=None) -> MatchingEngine:
    """Create the engine from settings (OpenAI + Qdrant in production)."""
    s = settings or get_settings()
    client = None
    if s.openai_api_key:
        client = OpenAIClient(s.openai_api_key, s.llm_model, s.embedding_model,
                              s.llm_timeout_s, s.llm_max_retries, make_cache(s))
    else:
        log.warning("OPENAI_API_KEY not set: using rule-based extraction and template explanations")

    extractor = OpenAIExtractor(client) if client else RuleBasedExtractor()
    embedder = OpenAIEmbedder(client) if client and s.embedding_provider == "openai" else HashingEmbedder()
    store = QdrantStore(s.qdrant_url, embedder.dim, s.qdrant_api_key) if s.vector_store == "qdrant" else InMemoryStore()
    return MatchingEngine(s, extractor, embedder, store, explainer=client)
