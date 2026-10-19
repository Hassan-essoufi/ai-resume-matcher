"""Explainable match score. No black box: every point of the score can be traced.

total = 100 * ( w_semantic   * semantic similarity of resume and job
              + w_required   * share of required skills covered
              + w_nice       * share of nice-to-have skills covered
              + w_experience * experience fit )

Skills match exactly after normalization ("k8s" == "kubernetes") or, if an
embedder is given, when their embeddings are very similar ("pytorch" ~ "deep learning frameworks").
"""
import numpy as np

from matcher.schemas import CandidateProfile, JobProfile, ScoreBreakdown


def rescale(cos: float, floor: float, ceiling: float) -> float:
    return float(np.clip((cos - floor) / (ceiling - floor), 0.0, 1.0))


def match_skills(wanted: list[str], have: list[str], embedder=None,
                 threshold: float = 0.8) -> tuple[list[str], list[str]]:
    """Return (matched, missing) for the wanted skills."""
    have_set = set(have)
    matched = [s for s in wanted if s in have_set]
    missing = [s for s in wanted if s not in have_set]
    if embedder is not None and missing and have:
        w = embedder.embed(missing)
        h = embedder.embed(list(have))
        sims = w @ h.T
        still_missing = []
        for i, skill in enumerate(missing):
            (matched if sims[i].max() >= threshold else still_missing).append(skill)
        missing = still_missing
    return matched, missing


def experience_fit(years: float, required: float) -> float:
    if required <= 0:
        return 1.0
    return float(min(1.0, years / required))


def score(job: JobProfile, cand: CandidateProfile, cosine: float, settings, embedder=None) -> ScoreBreakdown:
    s = settings
    semantic = rescale(cosine, s.semantic_floor, s.semantic_ceiling)

    m_req, miss_req = match_skills(job.required_skills, cand.skills, embedder, s.skill_similarity_threshold)
    m_nice, _ = match_skills(job.nice_to_have_skills, cand.skills, embedder, s.skill_similarity_threshold)
    req_cov = len(m_req) / len(job.required_skills) if job.required_skills else 1.0
    nice_cov = len(m_nice) / len(job.nice_to_have_skills) if job.nice_to_have_skills else 0.0
    exp = experience_fit(cand.years_experience, job.min_years_experience)

    # If the job lists no nice-to-have skills, give that weight to required skills
    w_req, w_nice = (s.w_required, s.w_nice) if job.nice_to_have_skills else (s.w_required + s.w_nice, 0.0)
    total = 100 * (s.w_semantic * semantic + w_req * req_cov + w_nice * nice_cov + s.w_experience * exp)

    return ScoreBreakdown(semantic=round(semantic, 3), required_coverage=round(req_cov, 3),
                          nice_coverage=round(nice_cov, 3), experience_fit=round(exp, 3),
                          matched_required=m_req, missing_required=miss_req, matched_nice=m_nice,
                          total=round(total, 1))
