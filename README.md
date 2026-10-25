# AI-Based Fake Review Detection — AI Review Checker

A local web app that keeps fake review detection in one place: paste any product review and instantly know if it's Fake or Genuine. Built for students, shoppers, and e-commerce trust.

Python + Scikit-learn + NLP on the backend, Streamlit on the frontend, and TF-IDF + Logistic Regression ( gpt-oss:20b-cloud style) for all AI features. There is no login - everything belongs to one local session.

🔗 Live Demo: https://ai-based-fake-review-detection-2j9wgy8slf8gl6cwmneqyx.streamlit.app/

### Features

| Area | What it does |
| :--- | :--- |
| Dashboard | Next prediction, fake vs real stats, recent checks. All numbers come from current session. Validated against your input. |
| Review Checker | Enter any product review, click Detect. Shows Fake or Genuine with confidence. Plans are marked saved once detected. |
| Text Cleaning | Lowercasing, punctuation removal, stopword removal, lemmatization. Colour-coded subjects, filtered before it's shown. |
| Model | TF-IDF + Logistic Regression. Trained on fake/real dataset. Accuracy ~90%. No API key needed, runs locally. |
| Simple UI | One page, one button. Paste review, get result. Works on mobile too. |
| Educational | Helps understand how fake reviews look and how NLP detects them. |

### Project structure
ai-based-fake-review-detection/
├── app.py              # main Streamlit app, all views
├── model.pkl           # trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # pip dependencies
├── style.css           # styling
└── README.md

Flask serves the frontend itself, so there is one process, one address.

### Quick start
backend + frontendpython -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501

### API overview

All responses look like {"result": "Fake", "confidence": 0.92}

- Dashboard: GET / - main page
- Detection: POST /predict - {"review": "text"}
- Cleaning: POST /api/clean - text preprocessing
- Model: POST /api/predict - returns Fake/Genuine

### Data

Local files studysphere style with model.pkl, vectorizer.pkl. No database. Every prediction is processed in memory. Nothing is stored server-side.

### Limits

- Model trained on English reviews only, Hindi reviews may give wrong results.
- Very short reviews (< 5 words) are rejected - too little context.
- AI output depends on the model. Invalid responses are rejected with error rather than replaced with made-up content.
- The browser loads Streamlit, so it needs internet. No authentication. Run locally and don't expose it to the internet as is.
