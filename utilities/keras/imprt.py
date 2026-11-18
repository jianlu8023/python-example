import os

os.environ["KERAS_BACKEND"] = "torch"
import json
import keras
from keras.api.models import model_from_json


def import_model_params(model: keras.Model):
    model.summary()


def import_model_from_structure_json(jsonpath: str) -> keras.Model:
    with open(jsonpath, 'r') as f:
        structure = json.load(f)
    return model_from_json(structure)
