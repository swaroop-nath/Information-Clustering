from typing import List, Dict
import manager as mgr
from numpy import log, sqrt
from nltk.tokenize import sent_tokenize, word_tokenize
from text_utils import simple_pre_process, rigorous_pre_process, tf_idf

def extract_top_k_paragraphs(text_doc: str, k: int = 5) -> (Dict[str, str], str):
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
    return sentence_mapper, mgr.paragraph_separator.join(ranked_paragraphs[:num_returns])

def _rank_and_order_paragraphs(processed_doc: str, tf_idf_values: Dict[str, float]) -> List[str]:
    '''
    This method is used to rank a paragraph based on normalized tf-idf and length scores.
    It returns all the paragraphs in ascending order of their rank.
    '''
    paragraph_score = {} # idx: score type dictionary. idx specifies the index of paragraph in the document.
    paragraphs = processed_doc.split(mgr.paragraph_separator)

    for idx, paragraph in enumerate(paragraphs):
        score = _find_paragraph_score(paragraph, tf_idf_values)
        paragraph_score[idx] = score

    ranked_paragraphs = [paragraphs[idx] for idx, _ in sorted(paragraph_score.items(), key=lambda item: item[1], reverse=True)]
    return ranked_paragraphs

def _find_paragraph_score(paragraph: str, tf_idf_values: Dict[str, float]) -> float:
    '''
    This method finds the score of a paragraph based on two parameters - normalized tf-idf
    and length score. The length score is determined by the sentences in the paragraph. The idea
    is that the marginal information added by incremental sentences to the paragraph decreases, hence
    a square root of the value is taken as an indicator.
    β_0(0.9) is chosen as the importance of normalized tf-idf and β_1(0.1) is chosen as the 
    importance of length.
    '''
    paragraph_length = 0
    tf_idf_score = 0
    beta_0 = 0.9
    beta_1 = 0.1
    sentences = sent_tokenize(paragraph)

    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        paragraph_length += len(word_tokens)
        for token in word_tokens:
            tf_idf_score += tf_idf_values[token]

    tf_idf_score = tf_idf_score / log(paragraph_length)
    length_score = sqrt(len(sentences))

    return beta_0 * tf_idf_score + beta_1 * length_score
        
