from typing import List
from text_utils import get_chunks

def extract_possible_path(cluster: List[str]) -> str:
    pass

def _construct_chunk_graph(cluster: List[str]) -> any:
    sentence_wise_chunks = []

    for sentence in cluster:
        chunks = get_chunks(sentence)
        sentence_wise_chunks.append(chunks)

    pass
