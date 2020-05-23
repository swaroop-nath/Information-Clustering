from typing import List, Dict
from paragraph_ranking import extract_top_k_paragraphs
from text_utils import find_abbreviations
import manager as mgr
from nltk import sent_tokenize

def summarize_docs(docs: List[str]) -> str:
    '''
    This method is the abstraction that is provided to the end user for utilization. It outputs a summary
    for a given list of documents.
    '''
    sentence_mapper = None
    important_paragraphs = []

    abbrevs = find_abbreviations(docs)

    for doc in docs:
        mapper, top_k = extract_top_k_paragraphs(doc, abbrevs)
        if sentence_mapper is None: sentence_mapper = mapper
        else: sentence_mapper.update(mapper)
        important_paragraphs.append(top_k)

    final_doc, final_sentence_mapper = _process_output(sentence_mapper, important_paragraphs)
    del sentence_mapper, important_paragraphs

    pass

def _process_output(sentence_mapper: Dict[str, str], important_paragraphs: List[str]) -> (str, Dict[str, str]):
    doc = mgr.paragraph_separator.join(important_paragraphs)
    final_mapper = {}

    for sentence in sent_tokenize(doc):
        final_mapper[sentence] = sentence_mapper[sentence]

    return doc, final_mapper
