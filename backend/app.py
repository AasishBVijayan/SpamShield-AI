from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Enron Spam Email Classifier API",
    description="A production-grade API backend to detect email spam using ML.",
    version="1.0.0"
)

# 2. Define the expected incoming data structure using Pydantic
class EmailPayload(BaseModel):
    subject: str
    message: str

import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded (will download silently if not present)
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

# 3. Load your saved model and vectorizer safely on startup
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

# 4. Create the core prediction endpoint
@app.post("/predict")
async def predict_spam(payload: EmailPayload):
    """
    Accepts an email subject and body, processes it, 
    and returns a spam/ham classification with confidence scores.
    """
    try:
        # Combine subject and message if your training script expected them joined
        full_text = f"{payload.subject} {payload.message}"
        
        # Clean and preprocess the input text exactly like we did during training!
        cleaned_text = transform_text(full_text)
        
        # Transform the cleaned text using your saved pipeline
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
        # Catch unexpected pipeline breaks and return a proper HTTP 500 status
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

# 5. Local development execution layer
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)