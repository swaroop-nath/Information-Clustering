import re
from spacy.lang.en.stop_words import STOP_WORDS as stop_words_list
import numpy as np
from typing import Dict, List
from normalise import normalise
import manager as mgr
from nltk.tokenize import sent_tokenize, word_tokenize

def simple_pre_process(text_doc: str) -> str:
    '''
    This method deals with removal of unnecessary portions in the text. It can arbitrary things as include exponent signs
    or square braces etc.
    Thing to be removed - 
    1. Quotation marks ("").
    2. References - usually of the form [1] or (1). In order to avoid omitting of years, like (1996), I will keep the
        length limited to 3.
    3. Non-English terms. For example - /mænˈdɛlə/ will be filtered out by the pre-processing.
    '''

    non_printable_chars = filter(lambda char_: char_ not in mgr.allowed_printable_chars, text_doc)
    non_printable_regex = []

    for char_ in non_printable_chars:
        non_printable_regex.append(mgr.prefix + char_ + mgr.suffix)
    
    quotation_free_doc = re.sub(mgr.quotation_regex, '', text_doc)
    reference_free_doc = re.sub(mgr.reference_regex, '', quotation_free_doc)

    final_doc = reference_free_doc

    for regex in non_printable_regex:
        final_doc = re.sub(regex, '', final_doc)

    return final_doc

def rigorous_pre_process(text_doc: str) -> (Dict[str, str], str):
    '''
    This is meant to be an extensive pre-processing procedure. It does assume that simple preprocessing has been done on the
    text. It further extends the pre-processing to lower-casing the text, removal of apostrophes'/expansion of abbreviations,
    like can't -> cannot, don't -> do not etc., lemmatizing the text, removal of stop-words etc.
    It returns a dictionary of sentences, with the processed sentence mapping to the raw sentence. This mapping can
    be used to output the un-processed sentence while outputting the summary. Along with that it also returns the 
    processed text document. This will be used for grouping paragraphs in the subsequent step.
    '''
    
    # Expanding contractions, like aren't to are not. And storing the unprocessed sentences.
    paragraphs = text_doc.split(mgr.paragraph_separator)
    unprocessed_sents = []
    expanded_paragraphs = []
    
    for paragraph in paragraphs:
        expanded_paragraph = _expand_textual_contractions(paragraph)
        for sentence in sent_tokenize(paragraph):
            unprocessed_sents.append(sentence)
        expanded_paragraphs.append(expanded_paragraph)
    del paragraphs

    # Treating abbreviations, like U.S.A --> United States of America.
    normalised_paragraphs = []
    for paragraph in expanded_paragraphs:
        normalised_paragraph = _normalize_paragraph(paragraph)
        normalised_paragraphs.append(normalised_paragraph)
    del expanded_paragraphs

    # Removing unwanted symbols.
    cleaned_paragraphs = []
    for paragraph in normalised_paragraphs:
        cleaned_paragraph = re.sub(mgr.unnecessary_identifier_regex, '', paragraph)
        cleaned_paragraphs.append(cleaned_paragraph)
    del normalised_paragraphs

    # Removing stop words and lower-casing the text
    stop_word_free_paragraphs = []
    for paragraph in cleaned_paragraphs:
        stop_word_free_paragraph = _remove_stop_words_and_lower_case(paragraph)
        stop_word_free_paragraphs.append(stop_word_free_paragraph)
    del cleaned_paragraphs

    # Lemmatize the text.
    lemmatized_paragraphs = []
    for paragraph in stop_word_free_paragraphs:
        lemmatized_paragraph = _lemmatize_paragraph(paragraph)
        lemmatized_paragraphs.append(lemmatized_paragraph)
    processed_text = mgr.paragraph_separator.join(lemmatized_paragraphs)
    del stop_word_free_paragraphs, lemmatized_paragraphs

    # Mapping to unprocessed sentences.
    sentence_mapper = _map_sentences(processed_text, unprocessed_sents)
    
    return (sentence_mapper, processed_text)

def _expand_textual_contractions(paragraph: str) -> str:
    expanded_sentences = []
    for sentence in sent_tokenize(paragraph):
        expanded_sentence = list(mgr.expander.expand_texts([sentence], precise=True))[0]
        expanded_sentences.append(expanded_sentence)
    
    processed_paragraph = mgr.sentence_separator.join(expanded_sentences)
    return processed_paragraph

def _normalize_paragraph(paragraph: str) -> str:
    normalised_sentences = []
    for sentence in sent_tokenize(paragraph):
        normalised_sentence = mgr.token_separator.join(normalise(word_tokenize(sentence), verbose=False, user_abbrevs=mgr.abbreviations_map))
        normalised_sentences.append(normalised_sentence)

    processed_paragraph = mgr.sentence_separator.join(normalised_sentences)
    return processed_paragraph

def _remove_stop_words_and_lower_case(paragraph: str) -> str:
    processed_sentences = []
    for sentence in sent_tokenize(paragraph):
        processed_words = [token.lower() for token in word_tokenize(sentence) if token not in stop_words_list]
        processed_sentence = mgr.token_separator.join(processed_words)
        processed_sentences.append(processed_sentence)
    
    processed_paragraph = mgr.sentence_separator.join(processed_sentences)
    return processed_paragraph

def _lemmatize_paragraph(paragraph: str) -> str:
    processed_sentences = []
    for sentence in sent_tokenize(paragraph):
        spacy_doc = mgr.spacy_tool(sentence)
        processed_tokens = []
        for token in spacy_doc:
            processed_tokens.append(token.lemma_)
        processed_sentences.append(mgr.token_separator.join(processed_tokens))

    processed_paragraph = mgr.sentence_separator.join(processed_sentences)
    return processed_paragraph

def _map_sentences(processed_doc: str, unprocessed_sents: List[str]) -> Dict[str, str]:
    read_count = 0
    paragraphs = processed_doc.split(mgr.paragraph_separator)
    mapper = {}

    for paragraph in paragraphs:
        for sentence in sent_tokenize(paragraph):
            mapper[sentence] = unprocessed_sents[read_count]
            read_count += 1
            if read_count == len(unprocessed_sents): break
        if read_count == len(unprocessed_sents): break
    return mapper

def tf_idf(text_doc: str) -> Dict[str, int]:
    '''
    This method is used to find out the tf_idf scores for each word in a document. Given a rigorously pre-processed
    document, this outputs a dictionary of words along with their tf-idf scores
    '''
    pass

def vectorize_sentence(sentence: str) -> np.ndarray:
    '''
    This method is responsible for mapping a given sentence to an embedding space. It outputs a numpy array that is the
    vector representation of the sentence.
    '''
    pass

def produce_chunk_graph(cluster: List[str]):
    pass
