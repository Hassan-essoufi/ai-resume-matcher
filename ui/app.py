"""Recruiter dashboard (Streamlit). Talks to the FastAPI service.

    streamlit run ui/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.getenv("API_KEY", "")}

st.set_page_config(page_title="Resume Matcher", page_icon="🎯", layout="wide")
st.title("🎯 AI Resume Matcher")
st.caption("Upload a job description and resumes, get a ranked, explained shortlist. "
           "Personal data is removed before any AI processing.")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"API online · {health['resumes']} resumes · LLM: {health['llm']}")
except Exception:
    st.sidebar.error(f"API not reachable at {API_URL}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Job description")
    job_file = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"], key="job")
with col2:
    st.subheader("2. Resumes")
    cv_files = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                                accept_multiple_files=True, key="cvs")
top_n = st.slider("Candidates to show", 3, 30, 10)

if st.button("Find best candidates", type="primary", disabled=not job_file):
    with st.spinner("Reading documents and extracting profiles..."):
        if cv_files:
            files = [("files", (f.name, f.getvalue())) for f in cv_files]
            r = requests.post(f"{API_URL}/resumes", files=files, headers=HEADERS, timeout=600)
            r.raise_for_status()
            for err in r.json()["errors"]:
                st.warning(f"{err['filename']}: {err['error']}")
        r = requests.post(f"{API_URL}/jobs", files={"file": (job_file.name, job_file.getvalue())},
                          headers=HEADERS, timeout=120)
        r.raise_for_status()
        job = r.json()
    with st.spinner("Ranking candidates..."):
        matches = requests.get(f"{API_URL}/jobs/{job['job_id']}/matches",
                               params={"top_n": top_n}, headers=HEADERS, timeout=600).json()
    st.session_state["result"] = (job, matches)

if "result" in st.session_state:
    job, matches = st.session_state["result"]
    p = job["profile"]
    st.divider()
    st.subheader(f"📋 {p['title']}")
    st.write("**Required:** " + ", ".join(p["required_skills"]))
    if p["nice_to_have_skills"]:
        st.write("**Nice to have:** " + ", ".join(p["nice_to_have_skills"]))
    st.write(f"**Experience:** {p['min_years_experience']:g}+ years · **Level:** {p['seniority']}")

    st.subheader("🏆 Ranked candidates")
    table = pd.DataFrame([{
        "Rank": m["rank"], "Candidate": m["filename"], "Score": m["score"],
        "Required skills": f"{len(m['breakdown']['matched_required'])}/{len(p['required_skills'])}",
        "Experience (yrs)": m["profile"]["years_experience"],
    } for m in matches])
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})

    for m in matches:
        b, e = m["breakdown"], m["explanation"] or {}
        with st.expander(f"#{m['rank']} · {m['filename']} · {m['score']:.0f}/100"):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.bar_chart(pd.Series({"Semantic": b["semantic"], "Required skills": b["required_coverage"],
                                        "Nice to have": b["nice_coverage"], "Experience": b["experience_fit"]}),
                             horizontal=True)
            with c2:
                if e:
                    st.markdown(f"**{e['summary']}**")
                    for s in e["strengths"]:
                        st.markdown(f"✅ {s}")
                    for g in e["gaps"]:
                        st.markdown(f"⚠️ {g}")
                if b["missing_required"]:
                    st.caption("Missing required: " + ", ".join(b["missing_required"]))
