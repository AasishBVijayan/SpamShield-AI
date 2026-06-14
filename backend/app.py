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