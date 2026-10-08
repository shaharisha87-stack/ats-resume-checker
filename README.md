# 🎯 Smart ATS Resume Analyzer & Job Matcher

A Streamlit app that uses Google Gemini to analyze a resume, estimate how well it would fare in an Applicant Tracking System (ATS), suggest STAR-style bullet rewrites, and recommend matching job roles.

> **Note:** the ATS score is an AI-generated estimate, not the output of a real ATS. It can vary slightly between runs. Use it as a guide to find gaps, not as a guarantee.

## Features

- **Resume parsing:** upload a `.pdf` (via `pypdf`) or `.txt` file (max 5 MB).
- **Mode A, Resume vs. Job Description:** paste a job posting to get a targeted score, matched and missing keywords (ranked Critical / Important / Nice-to-have), and tailored rewrites.
- **Mode B, General Audit:** leave the job description empty to get a standalone audit (structure, impact metrics, formatting) plus realistic matching roles.
- **Matching roles:** titles, seniority fit, skills you have vs. skills to learn, and next steps.
- **Dashboard:** metrics row, colored score bar, and three tabs (Score & Keywords, Improvements, Job Roles).
- **Export:** download the full analysis as a Markdown report.
- **Error handling:** empty, corrupt, encrypted or scanned PDFs, missing or invalid API keys, and quota or outage errors all show friendly messages.

## Project structure

```
ats-resume-analyzer/
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
├── DEPLOYMENT_GUIDE.md   # GitHub web UI + Streamlit Cloud steps
└── README.md
```

## Run locally

1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key at https://aistudio.google.com/apikey.
4. Provide the key in one of two ways:
   - **Secrets file:** create `.streamlit/secrets.toml` containing:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - **Sidebar:** paste the key into the sidebar field when the app runs.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

Add `.streamlit/secrets.toml` to `.gitignore` if you use git. **Never commit your API key.**

## Deploy

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for step-by-step instructions to publish on GitHub through the web UI and deploy free on Streamlit Community Cloud.

## Models

The app tries `gemini-3.8-flash` first, then `gemini-3.6-flash` and `gemini-3.5-flash-lite` if a model is unavailable, overloaded or rate-limited. Each model gets one retry on 500/503 errors.

Google retires Gemini models frequently (`gemini-2.0-flash` and new-user access to `gemini-2.5-flash` are already gone). If you get 404 errors, either edit `MODEL_CHAIN` at the top of `app.py`, or type a current model ID into the sidebar's **Model override** field. Check https://ai.google.dev/gemini-api/docs/models for current IDs.

## Limitations

- Scanned or image-only PDFs can't be read (real ATS software can't read them either). Export a text-based PDF instead.
- Very long inputs are truncated to about 30,000 characters.
- Output quality depends on the model. Check every suggested rewrite against your real experience, and replace `[X%]`-style placeholders with true numbers. Never claim results you can't back up.
- Free-tier Gemini quotas are limited and may block heavy use.

## Privacy

Resume and job-description text is sent to Google's Gemini API for processing. On Google's free API tier, submitted content may be used to improve their products. Remove sensitive personal details before uploading, and tell any other users of your deployment about this.

## Tech stack

[Streamlit](https://streamlit.io) · [google-genai](https://pypi.org/project/google-genai/) · [pypdf](https://pypi.org/project/pypdf/)

## License

Choose a license for your repository (for example MIT) and add a `LICENSE` file.


