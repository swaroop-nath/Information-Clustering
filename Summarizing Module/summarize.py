from typing import List, Dict
from paragraph_ranking import extract_top_k_paragraphs
from sentence_clustering import extract_clusters
from text_utils import find_abbreviations
import manager as mgr
from nltk import sent_tokenize
import logging

def summarize_docs(docs: List[str]) -> str:
    '''
    This method is the abstraction that is provided to the end user for utilization. It outputs a summary
    for a given list of documents.
    '''
    sentence_mapper = None
    important_paragraphs = []

    logging.info('method: summarize_docs- Finding abbreviations in the text')
    abbrevs = find_abbreviations(docs)

    logging.info('method: summarize_docs- Extracting top_k paragraphs from the document')
    for doc in docs:
        mapper, top_k = extract_top_k_paragraphs(doc, abbrevs)
        if sentence_mapper is None: sentence_mapper = mapper
        else: sentence_mapper.update(mapper)
        important_paragraphs.append(top_k)

    logging.info('method: summarize_docs- Post Processing the output of gathering important paragraphs')
    final_doc, final_sentence_mapper = _process_output(sentence_mapper, important_paragraphs)
    del sentence_mapper, important_paragraphs

    logging.info('method: summarize_docs- Starting clustering of sentences in important paragraphs')
    # extract_clusters(final_doc)
    pass

def _process_output(sentence_mapper: Dict[str, str], list_of_important_paragraphs: List[str]) -> (str, Dict[str, str]):
    doc = mgr.paragraph_separator.join(list_of_important_paragraphs)
    final_mapper = {}

    for imp_paragraphs in list_of_important_paragraphs:
        for paragraph in imp_paragraphs.split('\n\n'):
            for sentence in sent_tokenize(paragraph):
                final_mapper[sentence] = sentence_mapper[sentence]

    return doc, final_mapper
