import os
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

with open(os.path.join(os.getenv("MODEL_PATH"), "model.pkl"), "rb") as f:
    model = pickle.loads(f.read())

with open(os.path.join(os.getenv("MODEL_PATH"), "transformer.pkl"), "rb") as f:
    transformer = pickle.loads(f.read())

data_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


@app.get("/")
async def root():
    return "shalom"


@app.post("/diagnose/")
async def diagnose(features: Features):
    data = features.dict()
    df = pd.Series({col: data.get(col) for col in data_columns}).to_frame().T
    X = transformer.transform(df)
    y = model.predict_proba(X).round(2)[0][1]
    return {"disease_proba": y}


if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=8000, access_log=True)
