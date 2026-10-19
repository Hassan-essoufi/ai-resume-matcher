import re

SKILL_ALIASES: dict[str, list[str]] = {
    # languages
    "python": ["python3"],
    "java": [], "javascript": ["js", "ecmascript"], "typescript": ["ts"],
    "c++": ["cpp"], "c#": ["csharp"], "go": ["golang"], "rust": [], "scala": [], "r": [],
    "sql": ["t-sql", "pl/sql"], "bash": ["shell scripting"], "php": [], "kotlin": [],
    # data / ML
    "machine learning": ["ml"], "deep learning": ["dl"], "nlp": ["natural language processing"],
    "computer vision": ["image processing"], "llm": ["large language models", "llms"],
    "rag": ["retrieval-augmented generation", "retrieval augmented generation"],
    "prompt engineering": [], "langchain": [], "langgraph": [], "hugging face": ["huggingface", "transformers"],
    "pytorch": ["torch"], "tensorflow": ["tf"], "keras": [], "scikit-learn": ["sklearn", "scikit learn"],
    "xgboost": [], "lightgbm": [], "pandas": [], "numpy": [], "spacy": [], "opencv": [],
    "mlflow": [], "dvc": [], "mlops": [], "statistics": ["statistical analysis"],
    "time series": ["forecasting"], "data visualization": ["dataviz", "data viz"],
    "power bi": ["powerbi"], "tableau": [], "excel": ["ms excel", "microsoft excel"],
    "spark": ["pyspark", "apache spark"], "hadoop": [], "airflow": ["apache airflow"],
    "kafka": ["apache kafka"], "dbt": [], "etl": ["elt", "data pipelines"],
    "vector databases": ["vector database", "qdrant", "pinecone", "chromadb", "faiss", "weaviate"],
    # backend / web
    "fastapi": [], "flask": [], "django": [], "node.js": ["nodejs"], "express": [],
    "spring boot": [], "rest api": ["restful", "rest apis", "restful apis", "api design"],
    "graphql": [], "react": ["reactjs", "react.js"], "angular": [], "vue": ["vue.js", "vuejs"],
    "html": [], "css": ["tailwind", "sass"],
    # databases
    "postgresql": ["postgres"], "mysql": [], "mongodb": ["mongo"], "redis": [],
    "elasticsearch": [], "sqlalchemy": [], "neo4j": [],
    # cloud / devops
    "docker": [], "kubernetes": ["k8s"], "aws": ["amazon web services"], "gcp": ["google cloud"],
    "azure": ["microsoft azure"], "terraform": [], "ci/cd": ["cicd", "ci cd", "github actions", "gitlab ci", "jenkins"],
    "git": ["github", "gitlab"], "linux": [], "microservices": [],
    # methods / soft
    "agile": ["scrum", "kanban"], "unit testing": ["pytest", "tdd"],
    "communication": [], "leadership": ["team lead", "mentoring"],
}

_ALIAS_TO_CANONICAL = {alias: canon for canon, aliases in SKILL_ALIASES.items() for alias in aliases}
_ALIAS_TO_CANONICAL.update({canon: canon for canon in SKILL_ALIASES})


def normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", skill.strip().lower())
    s = s.strip(" .,;:()")
    return _ALIAS_TO_CANONICAL.get(s, s)


def normalize_skills(skills: list[str]) -> list[str]:
    seen, out = set(), []
    for skill in skills:
        n = normalize_skill(skill)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


# Longest aliases first so "deep learning" wins over "learning"-like overlaps
_TERMS = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)
_SHORT_AMBIGUOUS = {"r", "go", "ts", "tf", "ml", "dl", "js", "git", "css", "sql", "dbt"}


def find_skills(text: str) -> list[str]:
    """Vocabulary lookup in free text (used by the rule-based extractor)."""
    lower = text.lower()
    found = []
    for term in _TERMS:
        if term in _SHORT_AMBIGUOUS:
            # short tokens must appear as a standalone, clearly delimited word
            pattern = rf"(?<![\w.+#/-]){re.escape(term)}(?![\w+#/-])"
            if term in {"r", "go"} and not re.search(rf"(?:^|[,(/|;:]\s*){re.escape(term)}\s*(?:[,)/|;]|$)",
                                                     lower, re.MULTILINE):
                continue
        else:
            pattern = rf"(?<![\w]){re.escape(term)}(?![\w])"
        if re.search(pattern, lower):
            found.append(_ALIAS_TO_CANONICAL[term])
    return normalize_skills(found)
