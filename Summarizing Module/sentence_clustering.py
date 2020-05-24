from typing import List, Dict
import numpy as np
from nltk import sent_tokenize
from text_utils import vectorize_sentence

def extract_clusters(text_doc: str) -> List[List[str]]:
    '''
    This method takes in a text document and finds the clusters of sentences. It returns the set of clusters
    found.
    '''
    sentences = sent_tokenize(text_doc)
    vector_lookup_table = _vectorize_sentences(sentences)
    pass

def _vectorize_sentences(sentences: List[str]) -> Dict[str, np.ndarray]:
    '''
    This is responsible for converting all the sentences into vector notation,
    and return them in a form compatible for the next step, i.e., clustering.
    '''
    lookup_table = {}
    for sentence in sentences:
        sentence_vector = vectorize_sentence(sentence)
        lookup_table[sentence] = sentence_vector

    return lookup_table