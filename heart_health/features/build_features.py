from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from heart_health.entities.feature_params import FeatureParams

FullTransformer = Union[ColumnTransformer, Pipeline]


def make_imputer(strategy: str) -> SimpleImputer:
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return imputer


def impute_features(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    imputer = make_imputer(strategy)
    features_transformed = imputer.fit_transform(df)
    features_pandas = pd.DataFrame(
        features_transformed, columns=df.columns, index=df.index,
    )
    return features_pandas


def make_categorical_imputer() -> SimpleImputer:
    return make_imputer(strategy="most_frequent")


def make_numerical_features() -> SimpleImputer:
    return make_imputer(strategy="mean")


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = categorical_pipeline()
    pipeline.fit(categorical_df)
    return pipeline.transform(categorical_df)


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = numerical_pipeline()
    pipeline.fit(numerical_df)
    return pipeline.transform(numerical_df)


def numerical_pipeline() -> Pipeline:
    imputer = make_numerical_features()
    pipeline = Pipeline([("impute", imputer), ("scale", StandardScaler())])
    return pipeline


def categorical_pipeline() -> Pipeline:
    imputer = make_categorical_imputer()
    pipeline = Pipeline([("impute", imputer), ("ohe", OneHotEncoder())])
    return pipeline


def full_transformer(params: FeatureParams) -> FullTransformer:
    cat_pipeline = categorical_pipeline()
    num_pipeline = numerical_pipeline()

    cat_features = list(filter(lambda f: f not in params.features_to_drop, params.categorical_features))
    num_features = list(filter(lambda f: f not in params.features_to_drop, params.numerical_features))

    return ColumnTransformer([
        ("cat_pipeline", cat_pipeline, cat_features),
        ("num_pipeline", num_pipeline, num_features)
    ])


def make_features(
        transformer: FullTransformer,
        df: pd.DataFrame,
        params: FeatureParams,
        test_mode: bool = False,
) -> Tuple[np.ndarray, Optional[np.array]]:
    ready_features_df = transformer.transform(df)

    if test_mode:
        return ready_features_df, None
    else:
        target = df[params.target_col].to_numpy()
        return ready_features_df, target
