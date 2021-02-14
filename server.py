import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Features(BaseModel):
    sex: int
    cp: int
    fbs: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float


app = FastAPI()

with open("models/model.pkl", "rb") as f:
    model = pickle.loads(f.read())

with open("models/transformer.pkl", "rb") as f:
    transformer = pickle.loads(f.read())

data_columns = pd.read_csv("data/raw/heart.csv").columns


@app.post("/diagnose/")
async def diagnose(features: Features):
    data = features.dict()
    df = pd.Series({col: data.get(col) for col in data_columns}).to_frame().T
    X = transformer.transform(df)
    y = model.predict_proba(X).round(2)[0][1]
    return {"disease_proba": y}


if __name__ == '__main__':
    uvicorn.run("server:app", access_log=True)
