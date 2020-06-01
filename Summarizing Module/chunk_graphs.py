from typing import List, Dict
from text_utils import get_chunks, simple_pre_process, rigorous_pre_process, tf_idf, vectorize_sentence
from utils import find_distance
import manager as mgr
from nltk.tokenize import word_tokenize

class UndefinedChunkError(Exception):
    def __init__(self, message):
        self.message = message
    
    def get_message(self):
        return self.message

def extract_possible_path(cluster: List[str], abbrevs: List[str]) -> str:
    are_all_sentences_same = _check_for_similarity(cluster)
    if are_all_sentences_same: return cluster[0]

    chunk_graph, chunk_mapper, word_tf_idf = _construct_chunk_graph(cluster, abbrevs)
    possible_paths = _get_possible_paths(chunk_graph, chunk_mapper, word_tf_idf)
    
    return possible_paths

def _check_for_similarity(cluster: List[str]) -> str:
    for reader in range(len(cluster) - 1):
        similar = cluster[reader] == cluster[reader + 1]
        if not similar: return False

    return True

def _construct_chunk_graph(cluster: List[str], abbrevs: Dict[str, str]) -> (Dict[str, List[str]], Dict[str, List[str]], Dict[str, float]):
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
    flattend_chunks = [chunk for sentence in sentence_wise_chunks for chunk in sentence]
    chunk_mapper = {}
    processed_docs = []
    
    for chunk in flattend_chunks:
        simple_processed_doc = simple_pre_process(text_doc=chunk)
        mapper, rigorously_processed_doc = rigorous_pre_process(text_doc=simple_processed_doc, abbrevs=abbrevs, remove_stop_words=False)

        for chunk, processed_chunk in mapper.items():
            if chunk_mapper.get(chunk) is None: chunk_mapper[chunk] = [processed_chunk]
            else: chunk_mapper[chunk].append(processed_chunk)
        processed_docs.append(rigorously_processed_doc)
    
    for chunk, processed_chunks in chunk_mapper.items():
        chunk_mapper[chunk] = list(set(processed_chunks))

    text_doc = mgr.paragraph_separator.join(processed_docs)
    tf_idf_values = tf_idf(text_doc=text_doc)

    return chunk_mapper, tf_idf_values

def _chunk_sentences(cluster: List[str]) -> List[List[str]]:
    sentence_wise_chunks = []

    for sentence in cluster:
        chunks = get_chunks(sentence)
        sentence_wise_chunks.append(chunks)

    return sentence_wise_chunks

def _get_possible_paths(chunk_graph: Dict[str, List[str]], chunk_mapper: Dict[str, List[str]], tf_idf_values: Dict[str, float], k: int = 2) -> List[str]:
    global START_TOKEN
    global END_TOKEN
    possible_paths = {}
    beam = [(chunk, str(path_number)) for path_number, chunk in enumerate(chunk_graph[START_TOKEN])]
    reverse_chunk_mapper = _reverse_chunk_map(chunk_mapper)

    while len(beam) > 0:
        FLAG_END_TOKEN_CHILD = False
        chunk, path_id = beam.pop(0)
        if possible_paths.get(path_id) is None: possible_paths[path_id] = [chunk]

        # Checking for children nodes, and judging value of adding them to the path based on a heuristic value
        children_chunks = chunk_graph[chunk]
        if len(children_chunks) == 1: 
            # Only one chunk child, add it directly to the beam if it is not the end
            if children_chunks[0] == END_TOKEN: continue
            beam.append((children_chunks[0], path_id))
            continue

        # More than one children chunk
        heuristic_scores = {}
        for child_chunk in children_chunks:
            # Checking if the child node is the end node
            if child_chunk == END_TOKEN: 
                FLAG_END_TOKEN_CHILD = True
                continue
            try:
                heuristic_val = _get_heuristic_value(child_chunk, reverse_chunk_mapper, tf_idf_values, possible_paths[path_id])
            except: heuristic_scores[child_chunk] = float('inf')
            heuristic_scores[child_chunk] = heuristic_val

        reverse_sorted_chunks = sorted(heuristic_scores, key=heuristic_scores.get, reverse=True)

        for idx, chunk in enumerate(reverse_sorted_chunks):
            if idx < k:
                if idx == 0 and not FLAG_END_TOKEN_CHILD: beam.append((chunk, path_id))
                else: 
                    new_path_id = path_id + str(idx)
                    # Copying the previous path for the new diverged path
                    possible_paths[new_path_id] = possible_paths[path_id]
                    beam.append((chunk, new_path_id))

    return list(possible_paths.values())

def _reverse_chunk_map(chunk_map: Dict[str, List[str]]) -> Dict[str, str]:
    reverse_chunk_map = {}
    for chunk, processed_chunks in chunk_map.items():
        for processed_chunk in processed_chunks:
            reverse_chunk_map[processed_chunk] = chunk

    return reverse_chunk_map

def _get_heuristic_value(chunk: str, mapper: Dict[str, str], tf_idf_values: Dict[str, float], path: List[str]) -> float:
    processed_chunk = mapper.get(chunk)
    if processed_chunk is None: raise UndefinedChunkError('Chunk omitted during pre-processing.')

    vectorized_chunk = vectorize_sentence(processed_chunk)
    avg_tf_idf = sum(tf_idf_values.values()) / len(tf_idf_values)

    for word in processed_chunk:
        if tf_idf_values.get(word) is None:
            tf_idf_score += avg_tf_idf
        else: tf_idf_score += tf_idf_values[word]
    
    distances = []
    for prev_chunk in path:
        vectorized_prev_chunk = vectorize_sentence(prev_chunk)
        distance = find_distance(vectorized_chunk, vectorized_prev_chunk)
        distances.append(distance)

    heuristic_val = tf_idf_score + min(distances)

    return heuristic_val