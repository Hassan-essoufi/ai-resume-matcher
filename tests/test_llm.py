"""OpenAI client logic tested with a fake SDK (no network, no key)."""
from types import SimpleNamespace

from matcher.cache import RedisCache
from matcher.llm import OpenAIClient, _JobOut, _ResumeOut


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def parse(self, model, messages, response_format, temperature):
        self.calls += 1
        if response_format is _ResumeOut:
            parsed = _ResumeOut(skills=["Python", "K8s", "python3"], years_experience=-1, seniority="mid",
                                job_titles=["ML Engineer"], education="MSc", summary="ML engineer")
        else:
            parsed = _JobOut(title="ML", required_skills=["Python", "PyTorch"], nice_to_have_skills=["python", "AWS"],
                             min_years_experience=3, seniority="mid", summary="ML role")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))],
                               usage=usage)


class FakeEmbeddings:
    def __init__(self):
        self.inputs = []

    def create(self, model, input):
        self.inputs.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])


class FakeRedis:
    def __init__(self, fail=False):
        self.data, self.ttls, self.fail = {}, {}, fail

    def get(self, k):
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(k)

    def set(self, k, v, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.data[k], self.ttls[k] = v, ex

    def ping(self):
        return not self.fail


def make_client(tmp_path=None):
    c = object.__new__(OpenAIClient)
    c.completions = FakeCompletions()
    c.embeddings = FakeEmbeddings()
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=c.completions), embeddings=c.embeddings)
    c.model, c.embedding_model = "fake-model", "text-embedding-3-small"
    c.cache = RedisCache("redis://unused", client=FakeRedis())
    return c


def test_resume_extraction_normalizes_and_caches(tmp_path):
    c = make_client(tmp_path)
    p = c.extract_resume("some resume text")
    assert p.skills == ["python", "kubernetes"] and p.years_experience == 0
    c.extract_resume("some resume text")
    assert c.completions.calls == 1          # second call served from cache


def test_job_extraction_removes_duplicates_from_nice_to_have(tmp_path):
    job = make_client(tmp_path).extract_job("job text")
    assert job.required_skills == ["python", "pytorch"] and job.nice_to_have_skills == ["aws"]


def test_embeddings_are_cached(tmp_path):
    c = make_client(tmp_path)
    c.embed(["a", "bb"])
    c.embed(["a", "ccc"])
    assert c.embeddings.inputs == [["a", "bb"], ["ccc"]]


def test_redis_cache_roundtrip_with_ttl():
    fake = FakeRedis()
    cache = RedisCache("redis://unused", ttl_days=2, client=fake)
    k = cache.key("resume", "model", "text")
    assert cache.get(k) is None
    cache.set(k, {"skills": ["python"]})
    assert cache.get(k) == {"skills": ["python"]}
    assert fake.ttls["rm:" + k] == 2 * 86400


def test_redis_down_does_not_break_the_pipeline():
    cache = RedisCache("redis://unused", client=FakeRedis(fail=True))
    cache.set("k", 1)                      # no exception
    assert cache.get("k") is None and cache.ping() is False
