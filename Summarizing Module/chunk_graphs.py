from typing import List, Dict
from text_utils import get_chunks, simple_pre_process, rigorous_pre_process, tf_idf
import manager as mgr

def extract_possible_path(cluster: List[str], abbrevs: List[str]) -> str:
    are_all_sentences_same = _check_for_similarity(cluster)
    if are_all_sentences_same: return cluster[0]

    chunk_graph = _construct_chunk_graph(cluster)
    possible_paths = _get_possible_paths(chunk_graph, word_tf_idf)

def _check_for_similarity(cluster: List[str]) -> str:
    for reader in range(len(cluster - 1)):
        similar = cluster[reader] == cluster[reader + 1]
        if not similar: return False

    return True

def _get_tf_idf_values(cluster: List[str], abbrevs: List[str]) -> Dict[str, float]:
    text_doc = mgr.paragraph_separator.join(cluster)
    simple_preprocessed_doc = simple_pre_process(text_doc=text_doc)
    _, rigorously_preprocessed_doc = rigorous_pre_process(text_doc=simple_processed_doc, abbrevs=abbrevs)
    tf_idf_values = tf_idf(text_doc=rigorously_preprocessed_doc)
    
    return tf_idf_values

def _construct_chunk_graph(cluster: List[str]) -> Dict[str, List[str]]:
    sentence_wise_chunks = _chunk_sentences(cluster)
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

    return chunk_graph

def _chunk_sentences(cluster: List[str]) -> List[List[str]]:
    sentence_wise_chunks = []

    for sentence in cluster:
        chunks = get_chunks(sentence)
        sentence_wise_chunks.append(chunks)

    return sentence_wise_chunks

def _get_possible_paths(chunk_graph: Dict[str, List[str]], tf_idf_values: Dict[str, float]) -> List[str]:
    global START_TOKEN
    global END_TOKEN
    possible_paths = []
    beam = chunk_graph[START_TOKEN]

    while len(beam) > 0:
        pass