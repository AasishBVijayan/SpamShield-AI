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

# 3. Load your saved model and vectorizer safely on startup
try:
    # Adjust filenames if you named them differently
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model and Vectorizer loaded successfully into memory.")
except Exception as e:
    print(f"❌ Error loading serialized model artifacts: {e}")
    raise RuntimeError("System configuration error: Missing model files.")