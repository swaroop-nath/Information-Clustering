import re
from spacy.lang.en.stop_words import STOP_WORDS as stop_words_list
import numpy as np
from typing import Dict, List, Tuple
from normalise import normalise # Takes time to import and causes warning signals
import manager as mgr # Takes appreciable time
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from abbreviations import schwartz_hearst
from utils import load_sentence_vectorizer, load_text_chunker_model # Takes time
from copy import copy

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

    non_printable_chars = filter(lambda char_: char_ not in mgr.allowed_printable_chars and char_ not in mgr.accepted_apostrophe_symbols, text_doc)
    non_printable_regex = []

    for char_ in non_printable_chars:
        non_printable_regex.append(mgr.prefix + char_ + mgr.suffix)
    
    quotation_free_doc = re.sub(mgr.quotation_regex, '', text_doc)
    reference_free_doc = re.sub(mgr.reference_regex, '', quotation_free_doc)

    matches = re.findall(mgr.acronym_period_regex, text_doc)
    for match in matches:
        replace_string = re.sub('\.', '', match)
        match = re.sub('\.', '\\.', match)
        reference_free_doc = re.sub(match, replace_string, reference_free_doc)

    for symbol in mgr.ellipse_symbols:
        reference_free_doc = reference_free_doc.replace(symbol,'-')

    final_doc = reference_free_doc

    for regex in non_printable_regex:
        final_doc = re.sub(regex, '', final_doc)

    return final_doc

def rigorous_pre_process(text_doc: str, abbrevs: Dict[str, str], remove_stop_words: bool = True, chunk_graph_call: bool = False) -> (Dict[str, str], List[List[str]]):
    '''
    This is meant to be an extensive pre-processing procedure. It does assume that simple preprocessing has been done on the
    text. It further extends the pre-processing to lower-casing the text, removal of apostrophes'/expansion of abbreviations,
    like can't -> cannot, don't -> do not etc., lemmatizing the text, removal of stop-words etc.
    It returns a dictionary of sentences, with the processed sentence mapping to the raw sentence. This mapping can
    be used to output the un-processed sentence while outputting the summary. Along with that it also returns the 
    processed text document. This will be used for grouping paragraphs in the subsequent step.
    '''

    paragraphs = text_doc.split(mgr.paragraph_separator)
    sentence_mapper = {}
    processed_paragraphs = []

    for paragraph in paragraphs:
        sentences = tokenize_to_sentences(paragraph)
        processed_paragraph = []
        for sentence in sentences:
            sentence = str(sentence)
            # Removing all apostrophes from possessive nouns
            apostrophe_free_sentence = copy(sentence)
            for apostrophe_symbol in mgr.accepted_apostrophe_symbols:
                apostrophe_free_sentence = re.sub(apostrophe_symbol + 's', '', apostrophe_free_sentence)

            # Expanding textual contractions
            expanded_sentence = _expand_textual_contractions(apostrophe_free_sentence)

            # Removing unwanted abbreviations, those inside braces
            cleaned_sentence = _remove_unwanted_abbrev(expanded_sentence, abbrevs)

            # Treating abbreviations, like U.S.A --> United States of America, and also context based abbreviations
            normalised_sentence = _normalize_sentence(cleaned_sentence, abbrevs)

            # Removing unwanted symbols
            if not chunk_graph_call: cleaned_sentence = re.sub(mgr.unnecessary_identifier_regex, '', normalised_sentence)
            else: cleaned_sentence = normalised_sentence

            # Removing stop words and lower-casing the text
            stop_word_free_sentence = _remove_stop_words_and_lower_case(cleaned_sentence, remove_stop_words = False)

            # Lemmatize the text
            lemmatized_sentence = _lemmatize_sentence(stop_word_free_sentence)

            # Doing post cleaning, removing double - spaces, spaces before period etc.
            try: processed_sentence = _do_post_processing(lemmatized_sentence, chunk_graph_call)
            except:
                print('Hi')
            
            sentence_mapper[processed_sentence] = sentence
            processed_paragraph.append(processed_sentence)
        
        # processed_paragraph = mgr.sentence_separator.join(processed_paragraph)
        processed_paragraphs.append(processed_paragraph) 
    
    # processed_text = mgr.paragraph_separator.join(processed_paragraphs)
    return (sentence_mapper, processed_paragraphs)

def _resolve_pronouns(text_doc: str) -> str:
    spacy_doc = mgr.spacy_tool(text_doc)
    resolved_doc = spacy_doc._.coref_resolved
    
    return resolved_doc

def _remove_unwanted_abbrev(paragraph: str, abbrevs: Dict[str, str]) -> str:
    for abbrev in abbrevs.keys():
        candidate = mgr.prefix + abbrev + mgr.suffix
        paragraph = re.sub(candidate, '', paragraph)
    
    return paragraph

def _expand_textual_contractions(sentence: str) -> str:
    expanded_sentence = list(mgr.expander.expand_texts([sentence], precise=True))[0]
    return expanded_sentence

def _normalize_sentence(sentence: str, user_abbrevs: str) -> str:
    user_abbrevs.update(mgr.abbreviations_map)
    normalised_sentence = mgr.token_separator.join(normalise(word_tokenize(sentence), verbose=False, user_abbrevs=user_abbrevs))
    return normalised_sentence

def _remove_stop_words_and_lower_case(sentence: str, remove_stop_words: bool) -> str:
    if remove_stop_words: processed_words = [token.lower() for token in word_tokenize(sentence) if token not in stop_words_list]
    else: processed_words = [token.lower() for token in word_tokenize(sentence)]
    processed_sentence = mgr.token_separator.join(processed_words)
    return processed_sentence

def _lemmatize_sentence(sentence: str) -> str:
    spacy_doc = mgr.spacy_tool(sentence)
    processed_tokens = []
    for token in spacy_doc:
        lemma = token.lemma_ if token.lemma_ != mgr.unnecessary_unresolved_pron else token.lower_ 
        processed_tokens.append(lemma)
    processed_sentence = mgr.token_separator.join(processed_tokens)

    return processed_sentence

def _do_post_processing(text_doc: str, chunk_graph_call: bool) -> str:
    doc = re.sub(mgr.unnecessary_space, ' ', text_doc)
    doc = re.sub(mgr.unnecessary_space_period, '.', doc)
    doc = re.sub(mgr.unnecessary_apostrophe, '', doc)
    doc = re.sub(mgr.period_regex, '', doc)

    if chunk_graph_call: return doc

    if not re.match(mgr.valid_eos_token, doc[-1]): return doc + '.'
    else: return doc

def _map_sentences(processed_doc: str, unprocessed_sents: List[str]) -> Dict[str, str]:
    read_count = 0
    paragraphs = processed_doc.split(mgr.paragraph_separator)
    mapper = {}

    for paragraph in paragraphs:
        for sentence in tokenize_to_sentences(paragraph):
            mapper[sentence] = unprocessed_sents[read_count]
            read_count += 1
            if read_count == len(unprocessed_sents): break
        if read_count == len(unprocessed_sents): break
    return mapper

def tf_idf(paragraphs: List[List[str]]) -> Dict[str, float]:
    '''
    This method is used to find out the tf_idf scores for each word in a document. Given a rigorously pre-processed
    document, this outputs a dictionary of words along with their tf-idf scores
    '''
    tf_matrix = {}
    idf_matrix = {}

    for paragraph in paragraphs:
        for sentence in paragraph:
            for token in word_tokenize(sentence):
                if tf_matrix.get(token) is None: tf_matrix[token] = 1
                else: tf_matrix[token] += 1

    total_token_count = sum(list(tf_matrix.values()))
    for token in tf_matrix.keys():
        tf_matrix[token] /= total_token_count

    for token in tf_matrix.keys():
        count = 0
        for paragraph in paragraphs:
            str_paragraph = mgr.sentence_separator.join(paragraph)
            if token in str_paragraph: count += 1
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

def vectorize_sentence(sentence: str) -> np.ndarray:
    '''
    This method is responsible for mapping a given sentence to an embedding space. It outputs a numpy array that is the
    vector representation of the sentence.
    '''
    vectorizer = load_sentence_vectorizer()
    return vectorizer([sentence]).numpy()

def get_chunks(sentence: str) -> List[str]:
    '''
    This method is responsible for getting the chunks present in a sentence. It returns the chunks found.
    '''
    tagged_sentence = _get_pos_tags(sentence)
    text_chunker_model = load_text_chunker_model()

    chunk_tree = text_chunker_model.parse(tagged_sentence)

    chunks = []
    for chunk in chunk_tree:
        chunk_texts = []
        
        if type(chunk) is Tree:
            # This chunk is a phrase
            for token, _ in chunk:
                chunk_texts.append(token)
        else:
            # This chunk is a word
            chunk_texts.append(chunk[0])

        text = mgr.token_separator.join(chunk_texts)
        chunks.append(text)

    return chunks
    
def _get_pos_tags(sentence: str) -> List[Tuple[str, str]]:
    '''
    This method serves the purpose of extracting the parts-of-speech tags for all the
    words of a given sentence.
    '''
    spacy_doc = mgr.spacy_tool(sentence)

    tagged_sentence = []
    for token in spacy_doc:
        tagged_token = (token.text, token.tag_)
        tagged_sentence.append(tagged_token)

    return tagged_sentence

def tokenize_to_sentences(text_doc: str) -> str:
    return mgr.sentence_tokenizer.tokenize(text_doc)