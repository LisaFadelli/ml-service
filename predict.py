import joblib # LOAD THE SAVED MODEL
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app=FastAPI(title="Term deposit prediction")
model=joblib.load("model.joblib")

class Client(BaseModel): # INPUT SCHEMA
    age: int = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int = Field(ge=1, le=31)
    month: str
    campaign: int = Field(ge=1)
    pdays: int
    previous: int = Field(ge=0)
    poutcome: str

# HEALTH ENDPOINT
# It tells FastAPI: when someone sends a GET request to /health, call the function below
# The function returns a dictionary and FastAPI turns it into JSON
@app.get("/health")
def health():
    return {"status": "ok"}

# PREDICTION ENDPOINT
# post is used because the request carries data in its body
# client: Client -> FastAPI reads the JSON boy, validates it against the Client class, and only calls your function if it's valid

@app.post("/predict")
def predict(client: Client):
    X = pd.DataFrame([client.model_dump()]) # convert the object into a plain dictionary
    probability = float(model.predict_proba(X)[0, 1])
    return {"probability": probability, "subscribes": probability >= 0.5} #It returns the probability, plus a yes/no flag using a 0.5 cut-off.