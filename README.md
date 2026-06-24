# SpamShield-AI (Email Spam Detection)

A lightweight end-to-end project to detect **spam vs ham** in email text using classical ML (TF‑IDF + model) and deploy it via a **FastAPI** backend. A small **Chrome extension** can call the API locally to scan pasted email content.

---

## Features

- **ML inference API** built with **FastAPI**
- **Text preprocessing** with NLTK (tokenization, stopword removal, Porter stemming)
- **TF‑IDF vectorization** using a persisted vectorizer
- **Spam/Ham prediction** with confidence score
- **Chrome extension popup UI** to test predictions quickly

---

## Project Layout

- `backend/app.py` – FastAPI server + inference endpoint
- `backend/model.pkl` – serialized trained model
- `backend/vectorizer.pkl` – serialized TF‑IDF/Vectorizer
- `pipeline.ipynb` – training + evaluation notebook (EDA, modeling, serialization)
- `datasets/` – datasets used for training and combination
- `spam-extension/` – Chrome extension (popup UI + API calls)

---

## API Usage

### Start the backend

1. Install dependencies (recommended):

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Server docs:
- `http://127.0.0.1:8000/docs`

### Predict endpoint

`POST /predict`

Request body:

```json
{
  "subject": "<email subject>",
  "message": "<email body>"
}
```

Response:

```json
{
  "status": "success",
  "prediction": "Ham" | "Spam",
  "label_code": 0 | 1,
  "confidence_score": 0.0
}
```

---

## Chrome Extension

The extension is configured to call:
- `http://127.0.0.1:8000/predict`

### Load the extension (Chrome)

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `spam-extension/` folder

### Use

- Paste text into the popup textarea
- Click **Analyze Text**
- The extension displays **Ham/Spam** + confidence

---

## Model / Training Notes

Training is done in `pipeline.ipynb`:
- Load and combine datasets into `datasets/spam.csv`
- Preprocess text with NLTK + Porter stemming
- Vectorize with TF‑IDF (`max_features=3000`)
- Train and evaluate multiple candidate classifiers
- Serialize the selected model and vectorizer to:
  - `backend/model.pkl`
  - `backend/vectorizer.pkl`

---

## Requirements

Core dependencies:
- fastapi
- uvicorn[standard]
- scikit-learn
- pandas, numpy, joblib
- nltk

See `requirements.txt` for the full list.

---

## Disclaimer

Email spam detection models can produce false positives/negatives—always review critical messages.

---
