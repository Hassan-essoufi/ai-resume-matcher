import json
import logging
from typing import Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from matcher.cache import NullCache, RedisCache
from matcher.schemas import CandidateProfile, Explanation, JobProfile, ScoreBreakdown
from matcher.skills import normalize_skills

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class _ResumeOut(BaseModel):
    skills: list[str]
    years_experience: float
    seniority: Literal["junior", "mid", "senior", "lead"]
    job_titles: list[str]
    education: str
    summary: str


class _JobOut(BaseModel):
    title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_years_experience: float
    seniority: Literal["junior", "mid", "senior", "lead"]
    summary: str


class _ExplanationOut(BaseModel):
    strengths: list[str]
    gaps: list[str]
    summary: str


RESUME_SYSTEM = """You extract structured data from resumes for a recruiting tool.
- skills: technical skills, tools and methods explicitly present. Short lowercase names ("python", "docker", "power bi").
- years_experience: total professional experience in years; internships count half; 0 if none.
- seniority: junior (<2 years), mid (2-5), senior (5-10), lead (10+ or manages a team).
- job_titles: most recent titles first.
- summary: one neutral sentence about the professional profile.
Never output personal data (name, contact, age, gender, nationality, photo). Never invent facts."""

JOB_SYSTEM = """You extract structured requirements from job descriptions for a recruiting tool.
- required_skills: skills that are required / must-have.
- nice_to_have_skills: skills described as a plus, bonus, preferred or optional.
- min_years_experience: minimum years required (0 if not stated).
- Short lowercase skill names. Never invent requirements that are not in the text."""

EXPLAIN_SYSTEM = """You help a recruiter understand why a candidate matches a job.
Use ONLY the facts provided (job requirements, candidate profile, computed scores).
- strengths: 2-3 short, concrete points.
- gaps: 0-3 short points (missing required skills, less experience than asked).
- summary: one sentence recommendation (e.g. "Strong match, interview first").
Never mention or guess personal characteristics (gender, age, origin, name)."""


class OpenAIClient:
    def __init__(self, api_key: str, model: str, embedding_model: str,
                 timeout: float = 60, max_retries: int = 3, cache: RedisCache | NullCache | None = None):
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.embedding_model = embedding_model
        self.cache = cache or NullCache()

    # structured output
    def _parse(self, system: str, user: str, schema: type[T], kind: str) -> T:
        key = self.cache.key(kind, self.model, system, user)
        if (hit := self.cache.get(key)) is not None:
            return schema.model_validate(hit)

        completions = getattr(self.client.chat.completions, "parse", None) \
            or self.client.beta.chat.completions.parse
        resp = completions(
            model=self.model, temperature=0, response_format=schema,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise RuntimeError(f"LLM refused or returned no data: {getattr(msg, 'refusal', '')}")
        if resp.usage:
            log.info("llm call kind=%s tokens_in=%s tokens_out=%s", kind,
                     resp.usage.prompt_tokens, resp.usage.completion_tokens)
        self.cache.set(key, msg.parsed.model_dump())
        return msg.parsed

    def extract_resume(self, text: str) -> CandidateProfile:
        out = self._parse(RESUME_SYSTEM, f"<resume>\n{text[:15000]}\n</resume>", _ResumeOut, "resume")
        data = out.model_dump()
        data["skills"] = normalize_skills(data["skills"])
        data["years_experience"] = max(0.0, data["years_experience"])
        return CandidateProfile(**data)

    def extract_job(self, text: str) -> JobProfile:
        out = self._parse(JOB_SYSTEM, f"<job>\n{text[:15000]}\n</job>", _JobOut, "job")
        data = out.model_dump()
        data["required_skills"] = normalize_skills(data["required_skills"])
        data["nice_to_have_skills"] = [s for s in normalize_skills(data["nice_to_have_skills"])
                                       if s not in data["required_skills"]]
        data["min_years_experience"] = max(0.0, data["min_years_experience"])
        return JobProfile(**data)

    def explain(self, job: JobProfile, candidate: CandidateProfile, breakdown: ScoreBreakdown) -> Explanation:
        facts = json.dumps({
            "job": job.model_dump(),
            "candidate": candidate.model_dump(),
            "scores": breakdown.model_dump(),
        }, ensure_ascii=False)
        out = self._parse(EXPLAIN_SYSTEM, facts, _ExplanationOut, "explain")
        return Explanation(**out.model_dump())

    # embeddings 
    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self.cache.key("emb", self.embedding_model, t) for t in texts]
        results: list = [self.cache.get(k) for k in keys]
        todo = [i for i, r in enumerate(results) if r is None]
        for start in range(0, len(todo), 100):                     # batch API calls
            batch = todo[start:start + 100]
            resp = self.client.embeddings.create(model=self.embedding_model,
                                                 input=[texts[i][:8000] for i in batch])
            for i, item in zip(batch, resp.data):
                results[i] = item.embedding
                self.cache.set(keys[i], item.embedding)
        return results
