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
        
        # Transform the raw text using your saved pipeline
        vectorized_text = vectorizer.transform([full_text])
        
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