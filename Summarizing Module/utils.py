import configparser
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List
import numpy as np
from sklearn.neighbors import KernelDensity

model_paths = None
sent2vec_model = None

def _load_model_path_props():
    global model_paths
    config = configparser.RawConfigParser()
    config.read('models.properties')
    model_paths = dict(config.items('MODEL_PATHS'))

def load_sentence_vectorizer() -> keras.Model:
    if model_paths is None: _load_model_path_props()
    global sent2vec_model
    if sent2vec_model is not None: return sent2vec_model
    sent2vec_model = keras.models.load_model(model_paths['sentence-encoder-path'])
    return sent2vec_model

def find_clusters_and_noise(sentence_vector_lookup: Dict[int, np.ndarray]):
    '''
    This method finds the sentence clusters in the given data. It returns the clusters, and the
    noisy sentences (which don't belong to any cluster).
    One hyperparameter - ρ_0.
    ρ_0 defines the threshold density at any point below which, the point can be considered as noise.
    '''
    pass

def _estimate_density(sentence_vector_lookup: Dict[int, float]) -> Dict[int, float]:
    '''
    This estimates the kernel density of the given data. For now, it will use Gaussian
    kernel.
    '''
    # Fitting an estimator to the data
    vectors = np.array(list(sentence_vector_lookup.values))
    estimator = KernelDensity(kernel="gaussian", bandwidth=0.2)
    estimator = estimator.fit(vectors)

    density_estimates = {}

    for data_pt, vector in sentence_vector_lookup.items:
        density_estimates[data_pt] = estimator.score_samples(vector.reshape(1, -1))
    
    return density_estimates

def _find_delta(density_map: Dict[int, float], p: float = 0.5) -> Dict[int, float]:
    '''
    This is used to calculate the δ parameter of the clustering algorithm being used.
    One hyperparameter - p.
    The value p signifies the norm that will be used while calculating the distance metric.

    For example, p = 2 signifies an L2 norm.
    '''
    pass