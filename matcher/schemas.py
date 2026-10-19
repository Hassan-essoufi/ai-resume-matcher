from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["junior", "mid", "senior", "lead"]
SENIORITY_ORDER = ["junior", "mid", "senior", "lead"]


class CandidateProfile(BaseModel):
    """Structured view of a resume, extracted by the LLM (or rules)."""
    skills: list[str] = Field(description="Technical and professional skills, short names, e.g. 'python', 'docker'")
    years_experience: float = Field(ge=0, description="Total years of professional experience (internships count half)")
    seniority: Seniority = Field(description="Overall seniority level")
    job_titles: list[str] = Field(default_factory=list, description="Recent job titles")
    education: str = Field(default="", description="Highest degree and field, e.g. 'MSc Computer Science'")
    summary: str = Field(default="", description="One-sentence professional summary, no personal data")


class JobProfile(BaseModel):
    """Structured view of a job description."""
    title: str
    required_skills: list[str] = Field(description="Must-have skills, short names")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Optional / bonus skills")
    min_years_experience: float = Field(default=0, ge=0)
    seniority: Seniority = "mid"
    summary: str = Field(default="", description="One-sentence summary of the role")


class ScoreBreakdown(BaseModel):
    semantic: float                # 0-1, rescaled embedding similarity
    required_coverage: float       # 0-1, share of required skills the candidate has
    nice_coverage: float           # 0-1
    experience_fit: float          # 0-1
    matched_required: list[str]
    missing_required: list[str]
    matched_nice: list[str]
    total: float                   # 0-100


class Explanation(BaseModel):
    strengths: list[str] = Field(description="2-3 concrete strengths for this job")
    gaps: list[str] = Field(description="0-3 concrete gaps or risks")
    summary: str = Field(description="One sentence recommendation for the recruiter")


class MatchResult(BaseModel):
    resume_id: str
    filename: str
    rank: int
    score: float
    breakdown: ScoreBreakdown
    profile: CandidateProfile
    explanation: Explanation | None = None
