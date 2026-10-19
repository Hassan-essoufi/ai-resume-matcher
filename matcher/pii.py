import logging
import re
from functools import lru_cache

from matcher.skills import SKILL_ALIASES, normalize_skill

try:   # without Presidio, scrubbing falls back to regex only
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    AnalyzerEngine = None

log = logging.getLogger(__name__)

PLACEHOLDERS = {
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "PERSON": "<NAME>",
    "LOCATION": "<LOCATION>",
    "URL": "<URL>",
    "DATE_TIME": "<DATE>",
}
# LOCATION is left out: small NER models often tag tech words as places
PRESIDIO_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL"]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.IGNORECASE)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
# "Firstname Lastname" (2-4 capitalized words) alone on a line near the top
NAME_LINE = re.compile(r"^\s*([A-ZÀ-Ý][a-zà-ÿ'-]+|[A-ZÀ-Ý]{2,})(\s+([A-ZÀ-Ý][a-zà-ÿ'-]+|[A-ZÀ-Ý]{2,})){1,3}\s*$")

# Tech words that look like names and must never be removed
ALLOW_LIST = {"Flask", "Django", "Spark", "Kafka", "Jenkins", "Ruby", "Julia", "Rust", "Swift",
              "Scala", "Go", "Pandas", "Keras", "Airflow", "Tableau", "Excel", "Azure"}


def regex_scrub(text: str) -> str:
    text = EMAIL.sub(PLACEHOLDERS["EMAIL_ADDRESS"], text)
    text = URL.sub(PLACEHOLDERS["URL"], text)
    # Phones need 9+ digits, so date ranges like "2019 - 2023" are kept
    text = PHONE.sub(lambda m: PLACEHOLDERS["PHONE_NUMBER"]
                     if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), text)
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and NAME_LINE.match(lines[first]) \
            and not set(lines[first].split()) & ALLOW_LIST:
        lines[first] = PLACEHOLDERS["PERSON"]
    return "\n".join(lines)


@lru_cache
def _presidio():
    if AnalyzerEngine is None:
        raise RuntimeError("presidio is not installed")
    # Small spaCy model keeps the Docker image light
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()


def presidio_scrub(text: str) -> str:
    analyzer, anonymizer = _presidio()
    results = analyzer.analyze(text=text, language="en", entities=PRESIDIO_ENTITIES)
    results = [r for r in results if r.score >= 0.5
               and text[r.start:r.end] not in ALLOW_LIST
               and normalize_skill(text[r.start:r.end]) not in SKILL_ALIASES]
    operators = {e: OperatorConfig("replace", {"new_value": v}) for e, v in PLACEHOLDERS.items()}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


def scrub(text: str) -> str:
    """Presidio if available, then the regex pass as a safety net."""
    global _presidio_ok
    if _presidio_ok:
        try:
            text = presidio_scrub(text)
        except Exception as e:   # not installed or spaCy model missing
            _presidio_ok = False
            log.warning("Presidio unavailable (%s): using regex PII scrubbing only", e)
    return regex_scrub(text)


_presidio_ok = True
