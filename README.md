# Instructions

## 1. Model

```shell
python setup.py sdist
pip install dist/<something>.tar
train_pipeline configs/config.yaml
```

## 2. Prediction API

```shell
cp -r models/ predict/
cd predict
docker build -t <name>:<tag>
docker run -p 8000:8000 <name>:<tag>
```

## 3. Test

POST localhost:8000/diagnose/

body:
```json
{"age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0, "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0}
```
expected:
```json
{"disease_proba": 0.87}
```
