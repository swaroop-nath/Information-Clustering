from typing import List, Dict
from text_utils import get_chunks, simple_pre_process, rigorous_pre_process, tf_idf, vectorize_sentence
from utils import find_distance, load_language_model, compute_sentence_plausibility
import manager as mgr
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging
from copy import copy
from tqdm import tqdm

START_TOKEN = '<<START>>'
END_TOKEN = '<<END>>'

class UndefinedChunkError(Exception):
    def __init__(self, message):
        self.message = message
    
    def get_message(self):
        return self.message

def extract_possible_path(processed_cluster: List[str], mapper: Dict[str, str], abbrevs: List[str]) -> str:
    are_all_sentences_same = _check_for_similarity(processed_cluster)
    if are_all_sentences_same: return mapper[processed_cluster[0]]

    cluster = []
    for sentence in processed_cluster:
        cluster.append(mapper[sentence])

    mgr.logging.info('method: extract_possible_path- Constructing Chunk Graph')
    print('method: extract_possible_path- Constructing Chunk Graph')
    chunk_graph, chunk_mapper, word_tf_idf = _construct_chunk_graph(cluster, abbrevs)
    mgr.logging.info('method: extract_possible_path- Getting the possible paths in the Chunk Graph')
    print('method: extract_possible_path- Getting the possible paths in the Chunk Graph')
    possible_paths = _get_possible_paths(chunk_graph, chunk_mapper, word_tf_idf)
    del chunk_graph
    del chunk_mapper
    del word_tf_idf
    mgr.logging.info('method: extract_possible_path- Scoring each path based on a language model')
    print('method: extract_possible_path- Scoring each path based on a language model')
    filtered_path = _get_most_plausible_path(possible_paths)
    
    return filtered_path

def _check_for_similarity(cluster: List[str]) -> str:
    for reader in range(len(cluster) - 1):
        similar = cluster[reader] == cluster[reader + 1]
        if not similar: return False

    return True

def _construct_chunk_graph(cluster: List[str], abbrevs: Dict[str, str]) -> (Dict[str, List[str]], Dict[str, List[str]], Dict[str, float]):
    mgr.logging.info('method: _construct_chunk_graph- Extracting chunks for every sentence in the cluster')
    print('method: _construct_chunk_graph- Extracting chunks for every sentence in the cluster')
    sentence_wise_chunks = _chunk_sentences(cluster)
    mgr.logging.info('method: _construct_chunk_graph- Extracting tf-idf values for heuristic scoring')
    print('method: _construct_chunk_graph- Extracting tf-idf values for heuristic scoring')
    chunk_mapper, word_tf_idf = _get_tf_idf_values(sentence_wise_chunks, abbrevs)
    global START_TOKEN
    global END_TOKEN
    START_TOKEN = '<<START>>'
    END_TOKEN = '<<END>>'
    chunk_graph = {START_TOKEN: []}

    mgr.logging.info('method: _construct_chunk_graph- Chunk Graph Construction Start . . . . ')
    print('method: _construct_chunk_graph- Chunk Graph Construction Start . . . . ')
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
    mgr.logging.info('method: _construct_chunk_graph- Chunk Graph Construction done')
    print('method: _construct_chunk_graph- Chunk Graph Construction done')

    return chunk_graph, chunk_mapper, word_tf_idf

def _get_tf_idf_values(sentence_wise_chunks: List[str], abbrevs: Dict[str, str]) -> Dict[str, float]:
    flattend_chunks = [chunk for sentence in sentence_wise_chunks for chunk in sentence]
    chunk_mapper = {}
    processed_docs = []
    
    for chunk in flattend_chunks:
        simple_processed_doc = simple_pre_process(text_doc=chunk)
        mapper, rigorously_processed_doc = rigorous_pre_process(text_doc=simple_processed_doc, abbrevs=abbrevs, remove_stop_words=False, chunk_graph_call = True)

        for chunk, processed_chunk in mapper.items():
            if chunk_mapper.get(chunk) is None: chunk_mapper[chunk] = [processed_chunk]
            else: chunk_mapper[chunk].append(processed_chunk)
        processed_docs.extend(rigorously_processed_doc)
    
    for chunk, processed_chunks in chunk_mapper.items():
        chunk_mapper[chunk] = list(set(processed_chunks))

    tf_idf_values = tf_idf(paragraphs=processed_docs)

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
    path_word_count = {}
    beam = [(chunk, str(path_number)) for path_number, chunk in enumerate(chunk_graph[START_TOKEN])]
    reverse_chunk_mapper = _reverse_chunk_map(chunk_mapper)

    # loop_count = 1
    traversed_nodes = []
    mgr.logging.info('method: _get_possible_paths- Starting Beam Search . . . . ')
    print('method: _get_possible_paths- Starting Beam Search . . . . ')
    while len(beam) > 0:
        END_TOKEN_CHILD = -1
        chunk, path_id = beam.pop(0)
        traversed_nodes.append(chunk)
        if possible_paths.get(path_id) is None:
            # New path being created 
            possible_paths[path_id] = [chunk]
            path_word_count[path_id] = len(word_tokenize(chunk))
        else:
            is_duplicate_available = _check_path_traversed(possible_paths[path_id], chunk)
            path_word_count[path_id] += len(word_tokenize(chunk))
            if is_duplicate_available or path_word_count[path_id] > 25:
                # Caught in a cyclic loop
                del possible_paths[path_id]
                del path_word_count[path_id]
                continue
            possible_paths[path_id].append(chunk)

        # Checking for children nodes, and judging value of adding them to the path based on a heuristic value
        children_chunks = chunk_graph[chunk]
        if len(children_chunks) == 1: 
            # Only one chunk child, add it directly to the beam if it is not the end
            if children_chunks[0] == END_TOKEN: continue
            beam.append((children_chunks[0], path_id))
            continue

        # More than one children chunk
        heuristic_scores = {}
        for idx, child_chunk in enumerate(children_chunks):
            # Checking if the child node is the end node
            if child_chunk == END_TOKEN: 
                END_TOKEN_CHILD = idx
                continue
            try:
                heuristic_val = _get_heuristic_value(child_chunk, reverse_chunk_mapper, tf_idf_values, possible_paths[path_id])
            except UndefinedChunkError: 
                # The exception is thrown for chunks which had been omitted during pre-processing.
                # Judging from the code (keeping in mind that stop word removal has been set to false),
                # the removed chunk can either be puctuation mark (like , ; : etc) or
                # an abbreviation inside braces.
                heuristic_val = float('inf')
            heuristic_scores[child_chunk] = heuristic_val
        del children_chunks

        reverse_sorted_chunks = sorted(heuristic_scores, key=heuristic_scores.get, reverse=True)
        del heuristic_scores

        for idx, chunk in enumerate(reverse_sorted_chunks):
            if idx < k:
                if idx == END_TOKEN_CHILD: continue
                else: 
                    new_path_id = path_id + '-' + str(idx)
                    # Copying the previous path for the new diverged path
                    possible_paths[new_path_id] = copy(possible_paths[path_id])
                    path_word_count[new_path_id] = path_word_count[path_id]
                    beam.append((chunk, new_path_id))

    final_paths = []
    detokenizer = TreebankWordDetokenizer()
    for path in possible_paths.values():
        string_path = detokenizer.detokenize(path)
        if len(word_tokenize(string_path)) >= 8:
            final_paths.append(string_path)

    mgr.logging.info('method: _get_possible_paths- Beam Search over, extracted {} paths in the graph'.format(len(final_paths)))
    print('method: _get_possible_paths- Beam Search over, extracted {} paths in the graph'.format(len(final_paths)))

    return final_paths

def _reverse_chunk_map(chunk_map: Dict[str, List[str]]) -> Dict[str, str]:
    reverse_chunk_map = {}
    for chunk, processed_chunks in chunk_map.items():
        for processed_chunk in processed_chunks:
            reverse_chunk_map[processed_chunk] = chunk

    return reverse_chunk_map

def _get_heuristic_value(chunk: str, mapper: Dict[str, str], tf_idf_values: Dict[str, float], path: List[str]) -> float:
    processed_chunk = mapper.get(chunk)
    if processed_chunk is None: raise UndefinedChunkError('Chunk omitted during pre-processing.')

    vectorized_chunk = vectorize_sentence(processed_chunk)[0]
    avg_tf_idf = sum(tf_idf_values.values()) / len(tf_idf_values)

    tf_idf_score = 0
    for word in word_tokenize(processed_chunk):
        if tf_idf_values.get(word) is None:
            tf_idf_score += avg_tf_idf
        else: tf_idf_score += tf_idf_values[word]
    
    distances = []
    for prev_chunk in path:
        vectorized_prev_chunk = vectorize_sentence(prev_chunk)[0]
        distance = find_distance(vectorized_chunk, vectorized_prev_chunk)
        distances.append(distance)

    heuristic_val = tf_idf_score + min(distances)

    return heuristic_val

def _check_path_traversed(path: List[str], new_chunk: str) -> bool:
    tuple_to_be_checked = (path[-1], new_chunk)
    for chunk_one, chunk_two in zip(path[:-1], path[1:]):
        if chunk_one == tuple_to_be_checked[0] and chunk_two == tuple_to_be_checked[1]: return True
    
    return False

def _get_most_plausible_path(paths: List[str]) -> str:
    max_score = -float('inf')
    max_score_sentence = ''
    for sentence in tqdm(paths):
        plausibility_score = compute_sentence_plausibility(sentence)
        if plausibility_score > max_score:
            max_score = plausibility_score
            max_score_sentence = sentence

    return max_score_sentence
    