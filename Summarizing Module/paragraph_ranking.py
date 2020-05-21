from typing import List, Dict
from text_utils import simple_pre_process, rigorous_pre_process, tf_idf

def extract_top_k_paragraphs(text_doc: str, k: int = 5) -> (Dict[str, str], List[str]):
    '''
    This method is used to extract the top k paragraphs in the document. It returns a tuple
    of a sentence mapper and the list of top k paragraphs.
    The sentence mapper serves the purpose of maintaining the actual sentence present in the
    document, so that a coherent and human readable output can be produced
    '''
    simple_processed_doc = simple_pre_process(text_doc=text_doc)
    sentence_mapper, rigorously_processed_doc = rigorous_pre_process(text_doc=simple_processed_doc)
    tf_idf_values = tf_idf(text_doc=rigorously_processed_doc)

    ranked_paragraphs = _rank_and_order_paragraphs(rigorously_processed_doc, tf_idf_values)

    num_returns = min(k, len(ranked_paragraphs))
    return ranked_paragraphs[:num_returns]

def _rank_and_order_paragraphs(processed_doc, tf_idf_values) -> List[str]:
    '''
    This method is used to rank a paragraph based on normalized tf-idf and length scores.
    0.9 is chosen as the importance of normalized tf-idf and 0.1 is chosen as the 
    importance of length.
    It returns all the paragraphs in ascending order of their rank.
    '''
    return []