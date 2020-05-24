import re
from spacy.lang.en.stop_words import STOP_WORDS as stop_words_list
import numpy as np
from typing import Dict, List
from normalise import normalise
import manager as mgr
from nltk.tokenize import sent_tokenize, word_tokenize
from abbreviations import schwartz_hearst
from utils import load_sentence_vectorizer

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

def rigorous_pre_process(text_doc: str, abbrevs: Dict[str, str]) -> (Dict[str, str], str):
    '''
    This is meant to be an extensive pre-processing procedure. It does assume that simple preprocessing has been done on the
    text. It further extends the pre-processing to lower-casing the text, removal of apostrophes'/expansion of abbreviations,
    like can't -> cannot, don't -> do not etc., lemmatizing the text, removal of stop-words etc.
    It returns a dictionary of sentences, with the processed sentence mapping to the raw sentence. This mapping can
    be used to output the un-processed sentence while outputting the summary. Along with that it also returns the 
    processed text document. This will be used for grouping paragraphs in the subsequent step.
    '''
    
    # Resolving coreference
    resolved_doc = _resolve_pronouns(text_doc)

    # Expanding contractions, like aren't to are not. And storing the unprocessed sentences.
    paragraphs = resolved_doc.split(mgr.paragraph_separator)
    unprocessed_sents = []
    expanded_paragraphs = []
    
    for paragraph in paragraphs:
        expanded_paragraph = _expand_textual_contractions(paragraph)
        for sentence in sent_tokenize(paragraph):
            unprocessed_sents.append(sentence)
        expanded_paragraphs.append(expanded_paragraph)
    del paragraphs

    # Removing unwanted abbreviations, those inside braces
    cleaned_paragraphs = []
    for paragraph in expanded_paragraphs:
        cleaned_paragraph = _remove_unwanted_abbrev(paragraph, abbrevs)
        cleaned_paragraphs.append(cleaned_paragraph)
    del expanded_paragraphs

    # Treating abbreviations, like U.S.A --> United States of America, and also context based abbreviations
    normalised_paragraphs = []
    for paragraph in cleaned_paragraphs:
        normalised_paragraph = _normalize_paragraph(paragraph, abbrevs)
        normalised_paragraphs.append(normalised_paragraph)
    del cleaned_paragraphs

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

    # Doing post cleaning, removing double - spaces, spaces before period etc.
    processed_text = _do_post_processing(processed_text)

    # Mapping to unprocessed sentences.
    sentence_mapper = _map_sentences(processed_text, unprocessed_sents)
    
    return (sentence_mapper, processed_text)

def _resolve_pronouns(text_doc: str) -> str:
    spacy_doc = mgr.spacy_tool(text_doc)
    resolved_doc = spacy_doc._.coref_resolved
    
    return resolved_doc

def _remove_unwanted_abbrev(paragraph: str, abbrevs: Dict[str, str]) -> str:
    for abbrev in abbrevs.keys():
        candidate = mgr.prefix + abbrev + mgr.suffix
        paragraph = re.sub(candidate, '', paragraph)
    
    return paragraph

def _expand_textual_contractions(paragraph: str) -> str:
    expanded_sentences = []
    for sentence in sent_tokenize(paragraph):
        expanded_sentence = list(mgr.expander.expand_texts([sentence], precise=True))[0]
        expanded_sentences.append(expanded_sentence)
    
    processed_paragraph = mgr.sentence_separator.join(expanded_sentences)
    return processed_paragraph

def _normalize_paragraph(paragraph: str, user_abbrevs: str) -> str:
    normalised_sentences = []
    user_abbrevs.update(mgr.abbreviations_map)

    for sentence in sent_tokenize(paragraph):
        normalised_sentence = mgr.token_separator.join(normalise(word_tokenize(sentence), verbose=False, user_abbrevs=user_abbrevs))
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

def _do_post_processing(text_doc: str) -> str:
    doc = re.sub(mgr.unnecessary_space, ' ', text_doc)
    doc = re.sub(mgr.unnecessary_space_period, '.', doc)
    doc = re.sub(mgr.unnecessary_unresolved_pron, '', doc)
    doc = re.sub(mgr.unnecessary_apostrophe, '', doc)
    return doc

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

def tf_idf(text_doc: str) -> Dict[str, float]:
    '''
    This method is used to find out the tf_idf scores for each word in a document. Given a rigorously pre-processed
    document, this outputs a dictionary of words along with their tf-idf scores
    '''
    paragraphs = text_doc.split(mgr.paragraph_separator)
    tf_matrix = {}
    idf_matrix = {}

    for paragraph in paragraphs:
        for sentence in sent_tokenize(paragraph):
            for token in word_tokenize(sentence):
                if tf_matrix.get(token) is None: tf_matrix[token] = 1
                else: tf_matrix[token] += 1

    total_token_count = sum(list(tf_matrix.values()))
    for token in tf_matrix.keys():
        tf_matrix[token] /= total_token_count

    for token in tf_matrix.keys():
        count = 0
        for paragraph in paragraphs:
            if token in paragraph: count += 1
        idf_matrix[token] = np.log((len(paragraphs) / count) + 1)
    
    '''
    An extra term '1' is added because it is highly possible that a lot of words would be present in all of the
    paragraphs (mind that paragraph is used as document for inverse document frequency). 
    In such a case, irrespective of term frequency, a lot of words would have a zero score of tf-idf,
    which might me misleading, hence 1 is added in order to avoid a sparse tf-idf score vector for words.
    '''

    tf_idf_dict = {}
    for token in tf_matrix.keys():
        tf_idf_dict[token] = tf_matrix[token] * idf_matrix[token]
    return tf_idf_dict

def find_abbreviations(text_docs: List[str]) ->Dict[str, str]:
    '''
    This method is used to find the list of abbreviations in the document.
    It returns a dictionary of type - abbr. : full_form
    '''
    pairs = {}
    for doc in text_docs:
        found = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=doc, most_common_definition=True)
        pairs.update(found)
    return pairs
    pass

def vectorize_sentence(sentence: str) -> np.ndarray:
    '''
    This method is responsible for mapping a given sentence to an embedding space. It outputs a numpy array that is the
    vector representation of the sentence.
    '''
    vectorizer = load_sentence_vectorizer()
    return vectorizer([sentence]).numpy()

def produce_chunk_graph(cluster: List[str]):
    pass
