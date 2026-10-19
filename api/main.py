
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from matcher.config import get_settings
from matcher.engine import JOBS, RESUMES, MatchingEngine, build_engine
from matcher.ingestion import UnsupportedFileError
from matcher.schemas import CandidateProfile, JobProfile, MatchResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

MAX_FILE_MB = 5
ALLOWED = {".pdf", ".docx", ".txt"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = build_engine()
    log.info("engine ready")
    yield


app = FastAPI(title="Resume Matching API", version="1.0.0", lifespan=lifespan)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_engine(request: Request) -> MatchingEngine:
    return request.app.state.engine


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info("request id=%s %s %s -> %s in %.0fms", request_id, request.method,
             request.url.path, response.status_code, ms)
    return response

class JobOut(BaseModel):
    job_id: str
    profile: JobProfile


class ResumeOut(BaseModel):
    resume_id: str
    filename: str
    profile: CandidateProfile


class ResumeError(BaseModel):
    filename: str
    error: str


class ResumesOut(BaseModel):
    added: list[ResumeOut]
    errors: list[ResumeError]


async def _read(file: UploadFile) -> bytes:
    name = file.filename or "upload"
    if not any(name.lower().endswith(ext) for ext in ALLOWED):
        raise UnsupportedFileError(f"{name}: only PDF, DOCX or TXT files are accepted")
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise UnsupportedFileError(f"{name}: file larger than {MAX_FILE_MB} MB")
    return data


@app.get("/health")
def health(engine: MatchingEngine = Depends(get_engine)) -> dict:
    try:
        n_jobs = len(engine.store.list_ids(JOBS))
        n_resumes = len(engine.store.list_ids(RESUMES))
    except Exception as e:
        raise HTTPException(503, f"vector store unavailable: {e}")
    cache_ok = engine.explainer.cache.ping() if engine.explainer else None
    return {"status": "ok", "jobs": n_jobs, "resumes": n_resumes,
            "llm": "openai" if engine.explainer else "offline", "cache_ok": cache_ok}


@app.post("/jobs", response_model=JobOut, dependencies=[Depends(require_api_key)])
async def create_job(file: UploadFile = File(...), engine: MatchingEngine = Depends(get_engine)):
    try:
        job_id, profile = engine.add_job(file.filename, await _read(file))
    except UnsupportedFileError as e:
        raise HTTPException(422, str(e))
    return JobOut(job_id=job_id, profile=profile)


@app.post("/resumes", response_model=ResumesOut, dependencies=[Depends(require_api_key)])
async def upload_resumes(files: list[UploadFile] = File(...), engine: MatchingEngine = Depends(get_engine)):
    added, errors = [], []
    for f in files:                       # one bad file must not fail the whole batch
        try:
            rid, profile = engine.add_resume(f.filename, await _read(f))
            added.append(ResumeOut(resume_id=rid, filename=f.filename, profile=profile))
        except UnsupportedFileError as e:
            errors.append(ResumeError(filename=f.filename or "", error=str(e)))
        except Exception:
            log.exception("failed to process %s", f.filename)
            errors.append(ResumeError(filename=f.filename or "", error="processing failed"))
    return ResumesOut(added=added, errors=errors)


@app.get("/jobs/{job_id}/matches", response_model=list[MatchResult], dependencies=[Depends(require_api_key)])
def get_matches(job_id: str, top_n: int = Query(10, ge=1, le=100), explain: bool = True,
                engine: MatchingEngine = Depends(get_engine)):
    try:
        return engine.match(job_id, top_n=top_n, explain=explain)
    except KeyError:
        raise HTTPException(404, f"job {job_id} not found")


@app.delete("/resumes/{resume_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_resume(resume_id: str, engine: MatchingEngine = Depends(get_engine)):
    """Right to erasure (GDPR / loi 09-08): remove a candidate's data."""
    engine.store.delete(RESUMES, resume_id)
