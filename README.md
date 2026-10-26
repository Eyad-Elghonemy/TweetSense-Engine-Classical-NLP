<div align="center">

<img src="logo.svg" alt="TweetSense-Engine Logo" width="120" />

# TweetSense-Engine — Twitter Sentiment Analysis

**A lightweight, production-ready sentiment analysis API for tweets, built with FastAPI and scikit-learn.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Uvicorn](https://img.shields.io/badge/Server-Uvicorn-2E7D32?logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)

</div>

---

## What is TweetSense-Engine?

TweetSense-Engine is a REST API that analyzes the sentiment of tweets, classifying them as **Positive**, **Negative**, or **Neutral** in real time.

---

## 🧠 Two Models, One API

TweetSense-Engine ships with **two independent classifiers**, so you can pick the trade-off that fits your use case:

| | BOW Model | GloVe Model |
|---|---|---|
| **Approach** | Bag-of-Words vectorization + SVM | Pre-trained GloVe Twitter embeddings + SVM |
| **Endpoint** | `POST /predict` | `POST /predict/glove` |
| **Best for** | Fast, sparse, vocabulary-exact matching | Capturing semantic similarity between words |
| **Strength** | Lightweight, no external embedding download | Handles synonyms and unseen phrasing better |


Both models share the same text preprocessing pipeline (emoticon normalization, stopword removal, lemmatization) before vectorization, so predictions stay consistent across endpoints.

---

## Features

| Feature | Detail |
|---|---|
| 🚀 **Async FastAPI backend** | Production-grade, non-blocking request handling via Uvicorn |
| 🧠 **Dual classifiers** | Switch between BOW+SVM and GloVe+SVM per request |
| 🧹 **Custom text preprocessing** | Emoticon mapping, noise removal, lemmatization tuned for tweet-style text |
| 🔐 **API key authentication** | Every endpoint requires `X-API-Key` in the request header |
| 🗂️ **Batch prediction** | Predict sentiment for a list of texts in one request |
| 🌍 **CORS enabled** | Ready for cross-origin frontends out of the box |

---

## Model Details

Both classifiers are **SVM** models, differing only in how text is vectorized:

```
BOW path:    TextProcessor → CountVectorizer (BOW) → SVM
GloVe path:  TextProcessor → GloVe-Twitter-25 embeddings (averaged) → SVM
```

**Sentiment classes:** `Negative`, `Positive`, `Neutral`

See [`notebook.ipynb`](./notebook.ipynb) for the exploratory training experiments behind both models.

---

## Project Structure

```
.
├── main.py                       # FastAPI app — routes, auth, CORS
├── requirements.txt              # Pinned dependencies
├── .env.example                  # Environment variable template
├── notebook.ipynb                # Exploratory training notebook
└── SRC/
    ├── config.py                 # Env loading, model/artifact loading, constants
    ├── artifacts/
    │   ├── bow_vecorizer.pkl     # BOW vectorizer
    │   ├── bbow_svc.pkl          # BOW SVM model
    │   └── svc_glove_twitter.pkl # GloVe SVM model
    ├── utils/
    │   └── text_processor.py     # Text cleaning & preprocessing pipeline
    └── models/
        ├── inference.py          # TextClassifier / TextClassifierGlove logic
        └── schemas.py             # Pydantic request/response schemas
```

---

## Requirements

- Python 3.x
- See `requirements.txt` for pinned versions — key packages:
  - `fastapi==0.139.0`
  - `uvicorn==0.49.0`
  - `scikit-learn==1.7.1`
  - `python-dotenv==1.2.2`
  - `python-multipart==0.0.32`
  - `gensim`, `nltk`, `spaCy`

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
APP_NAME="TweetSense-Engine"
VERSION="1.0.0"
API_SECRET_KEY="your-strong-secret-key"
```

### 5. Add the model artifacts

Place the pre-trained model files at:

```
SRC/artifacts/bow_vecorizer.pkl
SRC/artifacts/bbow_svc.pkl
SRC/artifacts/svc_glove_twitter.pkl
```

> If these files are missing, the API will fail to start — both classifiers are loaded at startup.

---

## Running the Service

```bash
uvicorn main:app --reload
```

Available at: `http://127.0.0.1:8000`
Swagger UI: `http://127.0.0.1:8000/docs`

---

## API Reference

### `GET /`
Health check. Requires `X-API-Key` header.

**Response**
```json
{
  "App_Name": "TweetSense-Engine",
  "Version": "1.0.0",
  "Status": "Up & Running"
}
```

---

### `POST /predict`
Predicts sentiment using the **BOW + SVM** model.

**Body**
```json
{
  "texts": [
    "This is a great product!",
    "I hate this service"
  ]
}
```

**Response**
```json
{
  "predictions": [
    { "text": "This is a great product!", "sentiment": "Positive" },
    { "text": "I hate this service", "sentiment": "Negative" }
  ]
}
```

---

### `POST /predict/glove`
Predicts sentiment using the **GloVe + SVM** model.

**Body**
```json
{
  "texts": [
    "The product works as expected"
  ]
}
```

**Response**
```json
{
  "predictions": [
    { "text": "The product works as expected", "sentiment": "Neutral" }
  ]
}
```

**Error responses (all endpoints)**

| Status | Reason |
|---|---|
| `401` | Missing or invalid `X-API-Key` |
| `500` | An error occurred during prediction |

---

## Roadmap

- [ ] Lazy-load the GloVe embeddings instead of loading them at import time
- [ ] Add a unified `/predict` endpoint with a `model` query parameter instead of two separate routes
- [ ] Add confidence scores alongside the predicted sentiment label
- [ ] Add request rate limiting
- [ ] Dockerfile for one-command deployment

---

## Contributing

Issues and pull requests are welcome. If you're proposing a change to the text preprocessing pipeline (`text_processor.py`) or either model, please include before/after evaluation metrics so the impact is easy to review.

---

## Security Notes

- Every endpoint (except none — all are protected) requires a valid `X-API-Key` header.
- Never commit your `.env` file — it is excluded by `.gitignore`.
- CORS is currently open (`allow_origins=["*"]`). Restrict this in production.

---

<div align="center">
<sub>TweetSense-Engine · built with FastAPI &amp; scikit-learn</sub>
</div>
