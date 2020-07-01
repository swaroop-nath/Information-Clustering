from typing import List, Dict
import manager as mgr
from numpy import log, sqrt
from nltk.tokenize import word_tokenize
from text_utils import simple_pre_process, rigorous_pre_process, tf_idf

def extract_top_k_paragraphs(text_doc: str, abbrevs: Dict[str, str], k: int = 10) -> (Dict[str, str], List[List[str]]):
    '''
    This method is used to extract the top k paragraphs in the document. It returns a tuple
    of a sentence mapper and the list of top k paragraphs.
    The sentence mapper serves the purpose of maintaining the actual sentence present in the
    document, so that a coherent and human readable output can be produced
    '''
    simple_processed_doc = simple_pre_process(text_doc=text_doc)
    sentence_mapper, rigorously_processed_doc = rigorous_pre_process(text_doc=simple_processed_doc, abbrevs=abbrevs)
    tf_idf_values = tf_idf(paragraphs=rigorously_processed_doc)

    mgr.logging.info('method: extract_top_k_paragraphs- Starting the ranking of paragraphs')
    ranked_paragraphs = _rank_and_order_paragraphs(rigorously_processed_doc, tf_idf_values)
    mgr.logging.info('method: extract_top_k_paragraphs- Ranked top {} paragraphs in the document'.format(k))

    num_returns = min(k, len(ranked_paragraphs))
    return sentence_mapper, ranked_paragraphs[:num_returns]

def _rank_and_order_paragraphs(paragraphs: List[List[str]], tf_idf_values: Dict[str, float]) -> List[List[str]]:
    '''
    This method is used to rank a paragraph based on normalized tf-idf and length scores.
    It returns all the paragraphs in ascending order of their rank.
    '''
    paragraph_score = {} # idx: score type dictionary. idx specifies the index of paragraph in the document.

    for idx, paragraph in enumerate(paragraphs):
        score = _find_paragraph_score(paragraph, tf_idf_values)
        paragraph_score[idx] = score

    ranked_paragraphs = [paragraphs[idx] for idx, _ in sorted(paragraph_score.items(), key=lambda item: item[1], reverse=True)]
    return ranked_paragraphs

def _find_paragraph_score(paragraph: List[str], tf_idf_values: Dict[str, float]) -> float:
    '''
    This method finds the score of a paragraph based on two parameters - normalized tf-idf
    and length score. The length score is determined by the sentences in the paragraph. The idea
    is that the marginal information added by incremental sentences to the paragraph decreases, hence
    a square root of the value is taken as an indicator.
    Two hyperparameters - β_0 and β_1.
    β_0(0.9) is chosen as the importance of normalized tf-idf and β_1(0.1) is chosen as the 
    importance of length.
    '''
    paragraph_length = 0
    tf_idf_score = 0
    beta_0 = 0.9
    beta_1 = 0.1

    for sentence in paragraph:
        word_tokens = word_tokenize(sentence)
        paragraph_length += len(word_tokens)
        for token in word_tokens:
            tf_idf_score += tf_idf_values[token]
        
    tf_idf_score = tf_idf_score / log(paragraph_length + 1)
    length_score = sqrt(len(paragraph))

    return beta_0 * tf_idf_score + beta_1 * length_score
        
