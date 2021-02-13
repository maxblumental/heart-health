import json
import logging

import click

from heart_health.data import read_data, split_train_val_data
from heart_health.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from heart_health.features import make_features
from heart_health.features.build_features import full_transformer
from heart_health.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logging.basicConfig(level=logging.DEBUG)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logging.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logging.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logging.info(f"train_df.shape is {train_df.shape}")
    logging.info(f"val_df.shape is {val_df.shape}")
    transformer = full_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df, y=train_df[training_pipeline_params.feature_params.target_col])

    train_features, train_target = make_features(
        transformer, train_df, training_pipeline_params.feature_params, test_mode=False
    )

    logging.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features, val_target = make_features(
        transformer, val_df, training_pipeline_params.feature_params, test_mode=False
    )

    logging.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(model, val_features)

    metrics = evaluate_model(predicts, val_target)

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logging.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    path_to_transformer = serialize_model(
        transformer, training_pipeline_params.output_transformer_path
    )

    return path_to_model, path_to_transformer, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
