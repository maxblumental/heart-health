import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from heart_health.entities.train_params import TrainingParams
from heart_health.features.build_features import FullTransformer

SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
        features: np.ndarray, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=300, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: SklearnRegressionModel, features: np.ndarray
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def serialize_model(
        model: Union[SklearnRegressionModel, FullTransformer], output: str
) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
