import os
import string
import nltk
import joblib
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =====================================================================
# 1. Initialize FastAPI Application
# =====================================================================
app = FastAPI(
    title="Enron Spam Email Classifier API",
    description="A production-grade API backend to detect email spam using ML.",
    version="1.0.0"
)

# =====================================================================
# 2. Pydantic Data Structures
# =====================================================================
class EmailPayload(BaseModel):
    subject: str
    message: str

# =====================================================================
# 3. NLTK Text Processing Setup
# =====================================================================
# Ensure NLTK resources are downloaded silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words_set = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

def transform_text(text):
    tokens = nltk.word_tokenize(text.lower())
    cleaned_tokens = [
        ps.stem(token)
        for token in tokens
        if token.isalnum() and token not in stop_words_set and token not in punctuation_set
    ]
    return " ".join(cleaned_tokens)

# =====================================================================
# 4. Load ML Model Artifacts Safely
# =====================================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("[INFO] Model and Vectorizer loaded successfully into memory.")
except Exception as e:
    print(f"[ERROR] Error loading serialized model artifacts: {e}")
    raise RuntimeError("System configuration error: Missing model files.")

# =====================================================================
# 5. Core API Endpoints
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Landing page endpoint. Replaces the 404 on the root URL with a helper dashboard.
    """
    return """
    <html>
        <head>
            <title>Enron Spam Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }
                .container { max-width: 600px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; }
                a { color: #3498db; text-decoration: none; font-weight: bold; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📧 Enron Spam Email Classifier API</h1>
                <p>The backend inference engine is running successfully.</p>
                <p>To interactively test your model with payloads, head over to the API Documentation:</p>
                <p>👉 <a href="/docs">View Interactive Swagger UI (/docs)</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Handles automatic browser favicon requests silently to avoid console 404 errors.
    """
    return HTMLResponse(content='<link rel="icon" href="data:,">')

@app.post("/predict")
async def predict_spam(payload: EmailPayload):
    """
    Accepts an email subject and body, processes it, 
    and returns a spam/ham classification with confidence scores.
    """
    try:
        # Combine subject and message
        full_text = f"{payload.subject} {payload.message}"
        
        # Clean and preprocess text
        cleaned_text = transform_text(full_text)
        
        # Vectorize text pipeline
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Generate binary prediction (0 = Ham, 1 = Spam)
        prediction = int(model.predict(vectorized_text)[0])
        
        # Calculate prediction probability/confidence
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = float(probabilities[prediction])
        
        # Map binary label to string representation
        label_mapping = {0: "Ham", 1: "Spam"}
        
        return {
            "status": "success",
            "prediction": label_mapping[prediction],
            "label_code": prediction,
            "confidence_score": round(confidence, 4)
        }
        
    except Exception as e:
        # Catch pipeline exceptions and return an HTTP 500 status
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

# =====================================================================
# 6. Execution Layer
# =====================================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)