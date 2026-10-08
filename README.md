 AI Resume ATS Checker

A Streamlit app that uses Gemini 2.5 Flash to review a resume and provide an estimated ATS-readiness score, strengths, formatting issues, missing keywords, and practical improvement suggestions.

Features

Upload PDF, DOCX, or TXT resumes

Extract resume text automatically

Generate an estimated ATS score out of 100

Show an ATS breakdown for:

Readability

Keyword alignment

Section structure

Skills

Experience

Education

Identify ATS-unfriendly formatting/structure issues

Suggest specific resume improvements

Optionally compare the resume with a target job description

Use Gemini structured JSON output for reliable results

Ready for deployment on Streamlit Community Cloud

Important: ATS scores are estimates. Different Applicant Tracking Systems use different rules, so the score cannot guarantee that a resume will pass a real employer's screening system.

1. Project structure

ai-resume-ats-checker/
├── app.py
├── requirements.txt
└── README.md

2. Get a Gemini API key

Create a Gemini API key through Google AI Studio.

Do not put the API key directly into app.py, README.md, or GitHub.

For local testing, you can use an environment variable:

Windows PowerShell

$env:GEMINI_API_KEY="YOUR_API_KEY"
streamlit run app.py

You can also use Streamlit Secrets locally with:

.streamlit/
└── secrets.toml

secrets.toml:

GEMINI_API_KEY = "YOUR_API_KEY"

Do not upload secrets.toml to GitHub.

3. Install and run locally
Make sure Python is installed.

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

The app will open in your browser.

4. How the app works

Resume Upload
      ↓
PDF / DOCX / TXT text extraction
      ↓
Clean extracted text
      ↓
Optional Job Description
      ↓
Gemini 2.5 Flash
      ↓
Structured JSON analysis
      ↓
ATS Score + Breakdown
      ↓
Strengths + Issues + Improvements

The app does not claim to reproduce a proprietary ATS. Gemini evaluates ATS-readiness using common resume parsing and screening principles.

5. Push the project to GitHub using the UI

You can do this without Git commands.

Step 1 — Create a repository

Go to GitHub and sign in.

Click:

+ → New repository

Use a name such as:

ai-resume-ats-checker

Choose Public if you want the easiest Streamlit Community Cloud setup.

Click:

Create repository

Step 2 — Upload the three files

Inside your new repository:

Add file → Upload files

Upload:

app.py
requirements.txt
README.md

Then click:

Commit changes

Your repository should look like:

ai-resume-ats-checker
│
├── app.py
├── requirements.txt
└── README.md

Step 3 — Never upload your API key

Do not upload:

.streamlit/secrets.toml

with your real API key.

6. Deploy on Streamlit Community Cloud

Go to Streamlit Community Cloud and sign in with GitHub.

Click:

Create app

Select:

Repository: your-username/ai-resume-ats-checker

Branch: main

Main file path: app.py

Then open Advanced settings.

In Secrets, enter:

GEMINI_API_KEY = "YOUR_API_KEY"

Choose a supported Python version, preferably the same Python version you tested locally.

Click:

Deploy

Streamlit will install the packages from requirements.txt and start app.py.

Your app will receive a URL similar to:

https://your-app-name.streamlit.app

7. Updating the deployed app

When you change app.py:

Open your GitHub repository.

Open app.py.

Click the pencil/edit button.

Make your changes.

Click Commit changes.

Streamlit Community Cloud watches the GitHub repository and normally redeploys after repository changes.

8. Common problems

Gemini API key not found

Check that the secret is named exactly:

GEMINI_API_KEY = "YOUR_API_KEY"

ModuleNotFoundError

Make sure the package is listed in requirements.txt.

Then redeploy the app.

Resume text is empty

The PDF may be scanned/image-only. This version works best with text-based PDFs and DOCX files.

Gemini analysis fails

Check:

API key is valid

Gemini API access is available

The app has internet access

Your API quota has not been exceeded

9. Future improvements

Good next features for this project:

Resume vs job-description keyword matching

Downloadable improvement report

Resume rewriting suggestions

Skill-gap analysis

Job-specific resume recommendations

LinkedIn profile checklist

Before/after ATS score comparison

History of previous analyses

Better handling of scanned PDFs with OCR

License

You can use and modify this project for learning and portfolio purposes.
