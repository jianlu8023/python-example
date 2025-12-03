import os

os.environ["KERAS_BACKEND"] = "torch"

import keras


def export_model_params(model: keras.Model):
    model.summary()


def export_model_structure_tojson(model: keras.Model) -> str:
    return model.to_json()


def export_model_structure_json_to_file(model: keras.Model, filepath: str):
    with open(filepath, 'w') as f:
        f.write(export_model_structure_tojson(model))
