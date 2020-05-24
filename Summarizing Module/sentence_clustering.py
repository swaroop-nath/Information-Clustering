from typing import List, Dict
import numpy as np
from nltk import sent_tokenize
from text_utils import vectorize_sentence
from utils import find_clusters_and_noise

def extract_clusters(text_doc: str) -> List[List[str]]:
    '''
    This method takes in a text document and finds the clusters of sentences. It returns the set of clusters
    found.
    '''
    sentences = sent_tokenize(text_doc)
    vector_lookup_table, sentence_lookup_table = _vectorize_sentences(sentences)

    pass

def _vectorize_sentences(sentences: List[str]) -> (Dict[int, np.ndarray], Dict[int, str]):
    '''
    This is responsible for converting all the sentences into vector notation,
    and return them in a form compatible for the next step, i.e., clustering.
    '''
    lookup_table = {}
    sentence_lookup = {}
    for idx, sentence in enumerate(sentences):
        sentence_vector = vectorize_sentence(sentence)
        lookup_table[sentence] = sentence_vector
        sentence_lookup[idx] = sentence

    return lookup_table, sentence_lookup