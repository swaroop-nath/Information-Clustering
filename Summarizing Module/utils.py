import configparser
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from chunker import NGramTagChunker
from pickle import load

model_paths = None
constant_props = None
sent2vec_model = None
text_chunker_model = None
language_model = None
char2idx = None
SEQ_LEN = None

def _load_model_path_props():
    global model_paths
    config = configparser.RawConfigParser()
    config.read('models.properties')
    model_paths = dict(config.items('MODEL_PATHS'))

def _load_constant_props():
    global constant_props
    config = configparser.RawConfigParser()
    config.read('models.properties')
    constant_props = dict(config.items('CONSTANTS'))

def load_sentence_vectorizer() -> keras.Model:
    global model_paths
    global sent2vec_model
    if sent2vec_model is not None: return sent2vec_model
    if model_paths is None: _load_model_path_props()
    if sent2vec_model is not None: return sent2vec_model
    sent2vec_model = keras.models.load_model(model_paths['sentence-encoder-path'])
    return sent2vec_model

def load_text_chunker_model() -> NGramTagChunker:
    global model_paths
    global text_chunker_model
    if text_chunker_model is not None: return text_chunker_model
    if model_paths is None: _load_model_path_props()
    with open(model_paths['text-chunker-path'], 'rb') as file:
        text_chunker_model = load(file)
    
    return text_chunker_model

def load_language_model():
    global model_paths
    global constant_props
    global language_model
    if language_model is not None: return language_model
    if model_paths is None: _load_model_path_props()
    if constant_props is None: _load_constant_props()
    language_model = keras.models.load_model(model_paths['language-model-path'])
    _load_meta_data_for_language_model()

def _load_meta_data_for_language_model():
    global char2idx
    global SEQ_LEN
    meta_data_base_path = model_paths['language-model-meta-data-path']
    SEQ_LEN = int(constant_props['language-model-sequence-length'])
    char2idx_file_path = meta_data_base_path + '/char2idx.map'
    with open(char2idx_file_path, 'rb') as file:
        char2idx = load(file)

def find_clusters_and_noise(sentence_vector_lookup: Dict[int, np.ndarray]) -> (List[List[int]], List[int]):
    '''
    This method finds the sentence clusters in the given data. It returns the clusters, and the
    noisy sentences (which don't belong to any cluster).
    Three hyperparameters - 
    1. ρ_0
    2. γ_thres
    ρ_0 defines the threshold density at any point below which, the point can be considered as noise.
    γ_thres defines the region in the decision graph which contain cluster centres
    '''
    rho_0 = 0.80
    gamma_thres = 0.20

    density_estimates = _estimate_density(sentence_vector_lookup)
    delta_values = _find_delta(sentence_vector_lookup, density_estimates)
    
    max_density = max(list(density_estimates.values()))
    min_density = min(list(density_estimates.values()))
    max_delta = max(list(delta_values.values()))
    min_delta = min(list(delta_values.values()))

    density_estimates = {k: (v - min_density) / (max_density - min_density) for k, v in density_estimates.items()}
    delta_values = {k: (v - min_delta) / (max_delta - min_delta) for k, v in delta_values.items()}

    cluster_centres = []
    for data_pt in density_estimates.keys():
        density = density_estimates[data_pt]
        delta = delta_values[data_pt]

        if density * delta > gamma_thres:
            density_peak_pt = data_pt
            cluster_centres.append(density_peak_pt)
    
    if len(cluster_centres) == 0:
        # No clusters found, all sentences returned as noise
        return None, list(sentence_vector_lookup.keys())
    
    clusters = {center: [] for center in cluster_centres}
    noisy_data = []

    for data_pt, density in density_estimates.items():
        if density < rho_0: 
            noisy_data.append(data_pt)
            continue
        belonging_cluster_centre = _assign_cluster(cluster_centres, data_pt, sentence_vector_lookup)
        clusters[belonging_cluster_centre].append(data_pt)

    deletable = []
    for center, cluster in clusters.items():
        if len(cluster) == 0:
            # This means that this is a duplicate sentence, and has been
            # taken care of in some other cluster. Hence such clusters can be omitted
            deletable.append(center)

    for center in deletable:
        del clusters[center]

    return list(clusters.values()), noisy_data

def _estimate_density(sentence_vector_lookup: Dict[int, float]) -> Dict[int, float]:
    '''
    This estimates the kernel density of the given data. For now, it will use Gaussian
    kernel.
    '''
    # Fitting an estimator to the data
    vectors = np.array(list(sentence_vector_lookup.values()))
    estimator = KernelDensity(kernel="gaussian", bandwidth=0.2)
    estimator = estimator.fit(vectors)

    density_estimates = {}

    for data_pt, vector in sentence_vector_lookup.items():
        density_estimates[data_pt] = estimator.score_samples(vector.reshape(1, -1))
    
    return density_estimates

def _find_delta(sentence_vector_lookup: Dict[int, np.ndarray], density_map: Dict[int, float], p: float = 0.5) -> Dict[int, float]:
    '''
    This is used to calculate the δ parameter of the clustering algorithm being used.
    One hyperparameter - p.
    The value p signifies the norm that will be used while calculating the distance metric.

    For example, p = 2 signifies an L2 norm.

    It return a dictionary of δ values for each data point.
    '''
    delta_map = {}

    max_density_data_pt = max(density_map, key= lambda data_pt: density_map[data_pt])

    for data_pt_one, density_one in density_map.items():
        delta_array = []
        FLAG_MAX_DENSITY = False
        if density_one == density_map[max_density_data_pt]: FLAG_MAX_DENSITY = True

        for data_pt_two, density_two in density_map.items():
            if data_pt_one == data_pt_two: continue
            if density_one >= density_two and not FLAG_MAX_DENSITY: continue
            distance = find_distance(vector_one=sentence_vector_lookup[data_pt_one], 
                            vector_two=sentence_vector_lookup[data_pt_two], norm_raise=p)
            delta_array.append(distance)
        
        if FLAG_MAX_DENSITY: delta_map[data_pt_one] = max(delta_array)
        else: delta_map[data_pt_one] = min(delta_array)

    return delta_map

def _assign_cluster(cluster_centres: List[int], data_pt: int, vector_lookup_table: Dict[int, np.ndarray]) -> int:
    '''
    Assigning data points to cluster_centres obtained.
    '''
    distances = {}
    for center in cluster_centres:
        distance = find_distance(vector_lookup_table[center], vector_lookup_table[data_pt])
        distances[center] = distance

    min_dist = float('inf')
    min_dist_ctr = None

    for center, distance in distances.items():
        if distance < min_dist:
            min_dist = distance
            min_dist_ctr = center

    return min_dist_ctr

def find_distance(vector_one: np.ndarray, vector_two: np.ndarray, norm_raise: float = 0.5) -> float:
    '''
    This method calculates the distance based on the input norm order.
    '''
    difference = vector_one - vector_two
    distance = np.linalg.norm(difference, ord=norm_raise)
    
    return distance

def _convert_text_to_int(text_string: str) -> np.array:
    global char2idx
    text_as_int = []
    for c in text_string:
        if char2idx.get(c) is not None: text_as_int.append(char2idx[c])
    return np.array(text_as_int)

def _split_input_target(batch):
    input_text = batch[:-1]
    target_text = batch[-1]
    return input_text, target_text    

def compute_sentence_plausibility(sentence: str) -> float:
    global language_model
    global SEQ_LEN
    load_language_model()
    sentence_as_int_array = _convert_text_to_int(sentence)
    sentence_as_dataset = tf.data.Dataset.from_tensor_slices(sentence_as_int_array)
    batched_dataset = sentence_as_dataset.window(size=SEQ_LEN + 1, shift=1, drop_remainder=True)
    input_target_split_sentence = batched_dataset.flat_map(lambda window: window.batch(SEQ_LEN + 1)).map(_split_input_target)

    total_probability = 0
    count = 0

    for input_text, target_text in input_target_split_sentence.__iter__():
        predictions = language_model(tf.reshape(input_text, shape=(1, SEQ_LEN)))
        total_probability += predictions[0, SEQ_LEN - 1, target_text].numpy()
        count += 1

    return total_probability / count