import re
from typing import Protocol

from matcher.schemas import CandidateProfile, JobProfile
from matcher.skills import find_skills, normalize_skills


class Extractor(Protocol):
    def resume(self, text: str) -> CandidateProfile: ...
    def job(self, text: str) -> JobProfile: ...


class OpenAIExtractor:
    """Production extractor: OpenAI structured outputs (see llm.py)."""

    def __init__(self, client):
        self.client = client

    def resume(self, text: str) -> CandidateProfile:
        return self.client.extract_resume(text)

    def job(self, text: str) -> JobProfile:
        return self.client.extract_job(text)


# ------------------------------------------------------------------ rule-based
YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?|ans?)", re.IGNORECASE)
DATE_RANGE = re.compile(r"((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|now|current|aujourd'hui)",
                        re.IGNORECASE)
NICE_MARKERS = re.compile(r"nice to have|a plus|bonus|preferred|is a plus|appreciated|optional",
                          re.IGNORECASE)


def _seniority(years: float) -> str:
    return "junior" if years < 2 else "mid" if years < 5 else "senior" if years < 10 else "lead"


def _years_from_dates(text: str, current_year: int = 2026) -> float:
    total = 0
    for start, end in DATE_RANGE.findall(text):
        end_year = current_year if not end.isdigit() else int(end)
        total += max(0, end_year - int(start))
    return float(min(total, 40))


class RuleBasedExtractor:
    def resume(self, text: str) -> CandidateProfile:
        stated = [int(y) for y in YEARS.findall(text)]
        years = float(max(stated)) if stated else _years_from_dates(text)
        first_lines = [line.strip() for line in text.split("\n") if line.strip()][:3]
        title = next((line for line in first_lines if not line.startswith("<")), "")
        return CandidateProfile(skills=find_skills(text), years_experience=years,
                                seniority=_seniority(years), job_titles=[title] if title else [],
                                summary=title)

    def job(self, text: str) -> JobProfile:
        required, nice = [], []
        for line in text.split("\n"):
            (nice if NICE_MARKERS.search(line) else required).extend(find_skills(line))
        required = normalize_skills(required)
        nice = [s for s in normalize_skills(nice) if s not in required]
        stated = [int(y) for y in YEARS.findall(text)]
        years = float(min(stated)) if stated else 0.0
        title = next((line.strip() for line in text.split("\n") if line.strip()), "Job")
        return JobProfile(title=title[:120], required_skills=required, nice_to_have_skills=nice,
                          min_years_experience=years, seniority=_seniority(years), summary=title[:200])
