from typing import List
from collections import Counter
import numpy as np
import math
from lexrank import LexRank

def rank_sentences(cluster_paths: List[str], noise_sentences: List[str]) -> List[str]:
    final_sentences = cluster_paths.copy()
    final_sentences.extend(noise_sentences)
    del cluster_paths, noise_sentences

    ranker = LexRank(final_sentences)
    sentence_scores = ranker.rank_sentences(final_sentences)

    scored_sentences = {}
    for idx, score in enumerate(sentence_scores):
        scored_sentences[final_sentences[idx]] = score

    sorted_scored_sentences = sorted(scored_sentences, key=scored_sentences.get)
    return sorted_scored_sentences