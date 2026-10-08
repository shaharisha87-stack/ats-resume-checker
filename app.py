import io
import json
import os

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


st.set_page_config(
    page_title="AI Resume ATS Checker",
    page_icon="📄",
    layout="wide",
)

# Gemini model
MODEL_NAME = "gemini-3.8-flash-"


def get_api_key():
    """Read Gemini API key from Streamlit Secrets or environment variables."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None

    return key or os.getenv("GEMINI_API_KEY")


def extract_text(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]

        for table in document.tables:
            for row in table.rows:
                paragraphs.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        return "\n".join(paragraphs)

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")

    raise ValueError(
        "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
    )


def clean_text(text):
    """Clean extracted resume text."""
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Keep prompt size reasonable
    return text[:60000]


def clean_json_response(text):
    """Convert Gemini's response into a Python dictionary."""
    text = text.strip()

    # Remove Markdown code fences if Gemini adds them
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    # Find the JSON object if there is extra text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


def normalize_result(data):
    """Make sure the Gemini response has all required fields."""
    result = {
        "ats_score": 0,
        "summary": "",
        "strengths": [],
        "improvements": [],
        "missing_keywords": [],
        "formatting_issues": [],
        "section_feedback": {},
        "ats_breakdown": {
            "readability": 0,
            "keyword_alignment": 0,
            "section_structure": 0,
            "skills": 0,
            "experience": 0,
            "education": 0,
        },
    }

    result.update(data)

    # Keep score between 0 and 100
    try:
        result["ats_score"] = max(
            0, min(100, int(result["ats_score"]))
        )
    except Exception:
        result["ats_score"] = 0

    # Make sure lists are actually lists
    for key in [
        "strengths",
        "improvements",
        "missing_keywords",
        "formatting_issues",
    ]:
        if not isinstance(result[key], list):
            result[key] = []

    # Make sure dictionaries exist
    if not isinstance(result["section_feedback"], dict):
        result["section_feedback"] = {}

    if not isinstance(result["ats_breakdown"], dict):
        result["ats_breakdown"] = {}

    # Normalize breakdown scores
    breakdown_keys = [
        "readability",
        "keyword_alignment",
        "section_structure",
        "skills",
        "experience",
        "education",
    ]

    for key in breakdown_keys:
        try:
            value = int(result["ats_breakdown"].get(key, 0))
            result["ats_breakdown"][key] = max(0, min(100, value))
        except Exception:
            result["ats_breakdown"][key] = 0

    return result


def analyze_resume(resume_text, job_description):
    """Send resume to Gemini and receive ATS analysis."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)

    if job_description.strip():
        job_context = f"""
TARGET JOB DESCRIPTION:
{job_description[:20000]}
"""
    else:
        job_context = """
No job description was supplied.
Evaluate general ATS-readiness.
"""

    prompt = f"""
You are an expert ATS resume reviewer and career coach.

Analyze the resume below.

Important rules:
- Give an estimated ATS-readiness score from 0 to 100.
- Do not invent information about the candidate.
- Check readability, structure, standard sections, skills,
  experience, education, formatting, and keywords.
- If a job description is supplied, compare the resume with it.
- Identify useful missing keywords.
- Do not recommend keyword stuffing or false claims.
- Give practical and specific improvements.
- Keep the feedback concise.
- All scores must be integers from 0 to 100.

Return ONLY a valid JSON object.
Do not use Markdown.
Do not write anything before or after the JSON.

Use exactly this structure:

{{
  "ats_score": 0,
  "summary": "Short overall assessment",
  "strengths": [
    "Strength 1",
    "Strength 2",
    "Strength 3"
  ],
  "improvements": [
    "Improvement 1",
    "Improvement 2",
    "Improvement 3"
  ],
  "missing_keywords": [
    "keyword 1",
    "keyword 2"
  ],
  "formatting_issues": [
    "Formatting issue 1",
    "Formatting issue 2"
  ],
  "section_feedback": {{
    "Summary/Profile": "Feedback",
    "Experience": "Feedback",
    "Education": "Feedback",
    "Skills": "Feedback",
    "Projects": "Feedback",
    "Certifications": "Feedback"
  }},
  "ats_breakdown": {{
    "readability": 0,
    "keyword_alignment": 0,
    "section_structure": 0,
    "skills": 0,
    "experience": 0,
    "education": 0
  }}
}}

RESUME:
{resume_text}

{job_context}
"""

    # IMPORTANT:
    # We intentionally do NOT use response_schema or
    # response_mime_type here. Gemini returns normal text,
    # and our Python code converts it into JSON.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        data = clean_json_response(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned an invalid JSON response. "
            "Please try again."
        ) from exc

    return normalize_result(data)


def score_label(score):
    if score >= 85:
        return "Excellent ATS readiness"
    if score >= 70:
        return "Good ATS readiness"
    if score >= 55:
        return "Needs improvement"

    return "Major improvements recommended"


# -----------------------------
# USER INTERFACE
# -----------------------------

st.title("📄 AI Resume ATS Checker")

st.write(
    "Upload your resume to get an estimated ATS score, strengths, "
    "formatting issues, missing keywords, and practical improvements."
)


with st.sidebar:
    st.header("Settings")

    st.info(
        "Your Gemini API key is read from Streamlit Secrets. "
        "Do not put the key directly inside app.py or GitHub."
    )

    st.caption(f"Gemini model: {MODEL_NAME}")


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX and TXT.",
)


job_description = st.text_area(
    "Optional: paste the target job description",
    height=220,
    placeholder=(
        "Adding the job description makes keyword "
        "alignment more useful."
    ),
)


if uploaded_file:

    try:
        resume_text = clean_text(
            extract_text(uploaded_file)
        )

        if len(resume_text) < 100:

            st.warning(
                "Very little text could be extracted. "
                "If this is a scanned/image-only PDF, "
                "use a text-based PDF or DOCX."
            )

        else:

            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:8000])

            if st.button(
                "🔍 Analyze Resume",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Analyzing your resume with Gemini..."
                ):

                    try:

                        result = analyze_resume(
                            resume_text,
                            job_description,
                        )

                        st.session_state["analysis"] = result
                        st.session_state["filename"] = (
                            uploaded_file.name
                        )

                    except Exception as exc:

                        st.error(
                            f"Analysis failed: {exc}"
                        )

    except Exception as exc:

        st.error(
            f"Could not read this file: {exc}"
        )


result = st.session_state.get("analysis")


if result:

    st.divider()

    st.subheader(
        f"Results for "
        f"{st.session_state.get('filename', 'resume')}"
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Estimated ATS Score",
            f"{result['ats_score']}/100",
        )

    with col2:

        st.success(
            score_label(result["ats_score"])
        )

        st.write(result["summary"])


    # ATS BREAKDOWN

    st.subheader("📊 ATS Breakdown")

    breakdown = result["ats_breakdown"]

    cols = st.columns(3)

    items = [
        ("Readability", "readability"),
        ("Keyword Alignment", "keyword_alignment"),
        ("Section Structure", "section_structure"),
        ("Skills", "skills"),
        ("Experience", "experience"),
        ("Education", "education"),
    ]

    for i, (label, key) in enumerate(items):

        with cols[i % 3]:

            value = int(
                max(
                    0,
                    min(
                        100,
                        breakdown.get(key, 0),
                    ),
                )
            )

            st.progress(value / 100)

            st.caption(
                f"{label}: {value}/100"
            )


    left, right = st.columns(2)


    # STRENGTHS

    with left:

        st.subheader("✅ Strengths")

        for item in result["strengths"]:
            st.markdown(f"- {item}")


        st.subheader("⚠️ Formatting / ATS Issues")

        if result["formatting_issues"]:

            for item in result["formatting_issues"]:
                st.markdown(f"- {item}")

        else:

            st.write(
                "No major formatting issues were identified."
            )


    # IMPROVEMENTS

    with right:

        st.subheader("🚀 Improvements")

        for item in result["improvements"]:
            st.markdown(f"- {item}")


        st.subheader("🔑 Missing Keywords")

        if result["missing_keywords"]:

            st.write(
                ", ".join(
                    result["missing_keywords"]
                )
            )

        else:

            st.write(
                "No specific missing keywords were identified."
            )


    # SECTION FEEDBACK

    st.subheader("🧩 Section Feedback")

    for section, feedback in result[
        "section_feedback"
    ].items():

        st.markdown(
            f"**{section}:** {feedback}"
        )


    st.caption(
        "Note: This is an AI-generated ATS-readiness "
        "estimate. Real ATS systems differ, and a score "
        "cannot guarantee that an application will pass "
        "screening."
    )

else:

    st.info(
        "Upload a resume and click "
        "“Analyze Resume” to see your results."
    )



       


 
       
   
     
   
