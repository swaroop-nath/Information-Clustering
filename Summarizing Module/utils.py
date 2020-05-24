import configparser
import tensorflow as tf
from tensorflow import keras
from typing import Dict

model_paths = None

def _load_model_path_props():
    global model_paths
    config = configparser.RawConfigParser()
    config.read('models.properties')
    model_paths = dict(config.items('MODEL_PATHS'))

def load_sentence_vectorizer() -> keras.Model:
    if model_paths is None: _load_model_path_props()
    model = keras.models.load_model(model_paths['sentence-encoder-path'])
    return model
