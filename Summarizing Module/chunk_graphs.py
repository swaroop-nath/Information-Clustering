from typing import List, Dict
from text_utils import get_chunks, simple_pre_process, rigorous_pre_process, tf_idf, vectorize_sentence
from utils import find_distance
import manager as mgr
from nltk.tokenize import word_tokenize

def extract_possible_path(cluster: List[str], abbrevs: List[str]) -> str:
    are_all_sentences_same = _check_for_similarity(cluster)
    if are_all_sentences_same: return cluster[0]

    chunk_graph, chunk_mapper, word_tf_idf = _construct_chunk_graph(cluster, abbrevs)
    possible_paths = _get_possible_paths(chunk_graph, chunk_mapper, word_tf_idf)

def _check_for_similarity(cluster: List[str]) -> str:
    for reader in range(len(cluster - 1)):
        similar = cluster[reader] == cluster[reader + 1]
        if not similar: return False

    return True

def _construct_chunk_graph(cluster: List[str], abbrevs: Dict[str, str]) -> (Dict[str, List[str]], Dict[str, str], Dict[str, float]):
    sentence_wise_chunks = _chunk_sentences(cluster)
    chunk_mapper, word_tf_idf = _get_tf_idf_values(sentence_wise_chunks, abbrevs)
    global START_TOKEN
    global END_TOKEN
    START_TOKEN = '<<START>>'
    END_TOKEN = '<<END>>'
    chunk_graph = {START_TOKEN: []}

    for chunked_sentence in sentence_wise_chunks:
        chunk_graph[START_TOKEN].append(chunked_sentence[0])
        for chunk_idx in range(len(chunked_sentence) - 1):
            if chunk_graph.get(chunked_sentence[chunk_idx]) is None: 
                chunk_graph[chunked_sentence[chunk_idx]] = [chunked_sentence[chunk_idx + 1]]
            else:
                chunk_graph[chunked_sentence[chunk_idx]].append(chunked_sentence[chunk_idx + 1])
        if chunk_graph.get(chunked_sentence[-1]) is None: 
            chunk_graph[chunked_sentence[-1]] = [END_TOKEN]
        else: chunk_graph[chunked_sentence[-1]].append(END_TOKEN)
    
    for start_node, children_nodes in chunk_graph.items():
        chunk_graph[start_node] = list(set(children_nodes))

    return chunk_graph, chunk_mapper, word_tf_idf

def _get_tf_idf_values(sentence_wise_chunks: List[str], abbrevs: Dict[str, str]) -> Dict[str, float]:
    flattend_doc = [chunk for sentence in sentence_wise_chunks for chunk in sentence]
    chunk_mapper = {}
    tf_idf_values = {}
    
    for chunk in flattend_doc:
        simple_processed_doc = simple_pre_process(text_doc=chunk)
        mapper, rigorously_preprocessed_doc = rigorous_pre_process(text_doc=simple_processed_doc, abbrevs=abbrevs)
        values = tf_idf(text_doc=rigorously_preprocessed_doc)

        chunk_mapper.update(mapper)
        tf_idf_values.update(values)
    
    return chunk_mapper, tf_idf_values

def _chunk_sentences(cluster: List[str]) -> List[List[str]]:
    sentence_wise_chunks = []

    for sentence in cluster:
        chunks = get_chunks(sentence)
        sentence_wise_chunks.append(chunks)

    return sentence_wise_chunks

def _get_possible_paths(chunk_graph: Dict[str, List[str]], chunk_mapper: Dict[str, str], tf_idf_values: Dict[str, float], k: int = 2) -> List[str]:
    global START_TOKEN
    global END_TOKEN
    possible_paths = []
    beam = [(chunk, path_number) for path_number, chunk in enumerate(chunk_graph[START_TOKEN])]
    reverse_chunk_mapper = {v: k for k, v in chunk_mapper.items()}

    while len(beam) > 0:
        chunk, path_id = beam.pop(0)
        processed_chunk = reverse_chunk_mapper[chunk]

        for word in word_tokenize(processed_chunk):
            if tf_idf_values.get(word) is None:
                tf_idf_score += 0.40
            else: tf_idf_score += tf_idf_values[word]

        distance_scores = []
        vectorized_chunk = vectorize_sentence(chunk)
        for prev_chunk in possible_paths[path_id]:
            vectorized_repr = vectorize_sentence(prev_chunk)
            distance = find_distance(vectorized_chunk, vectorized_repr)
            # distance_scores