from typing import List, Dict
import numpy as np
from text_utils import vectorize_sentence
from utils import find_clusters_and_noise
import logging

def extract_clusters(paragraphs: List[List[str]]) -> (List[List[str]], List[str]):
    '''
    This method takes in a text document and finds the clusters of sentences. It returns the set of clusters
    found.
    '''
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    logging.info('method: extract_clusters- Vectorizing sentences')
    vector_lookup_table, sentence_lookup_table = _vectorize_sentences(sentences)

    logging.info('method: extract_clusters- Find clusters in the document')
    clusters, noisy_data = find_clusters_and_noise(vector_lookup_table)

    processed_clusters = []
    processed_noise = []

    for cluster in clusters:
        temporary = []
        for data_pt in cluster:
            temporary.append(sentence_lookup_table[data_pt])
        processed_clusters.append(temporary)

    for noise in noisy_data:
        processed_noise.append(sentence_lookup_table[noise])

    return processed_clusters, processed_noise

def _vectorize_sentences(sentences: List[str]) -> (Dict[int, np.ndarray], Dict[int, str]):
    '''
    This is responsible for converting all the sentences into vector notation,
    and return them in a form compatible for the next step, i.e., clustering.
    '''
    lookup_table = {}
    sentence_lookup = {}
    for idx, sentence in enumerate(sentences):
        sentence_vector = vectorize_sentence(sentence)
        lookup_table[idx] = sentence_vector[0]
        sentence_lookup[idx] = sentence

    return lookup_table, sentence_lookup