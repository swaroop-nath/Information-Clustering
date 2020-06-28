from typing import List, Dict
from paragraph_ranking import extract_top_k_paragraphs
from sentence_clustering import extract_clusters
from text_utils import find_abbreviations, rigorous_pre_process, tokenize_to_sentences
from sentence_ranking import rank_sentences
from chunk_graphs import extract_possible_path
import manager as mgr
import logging
from time import time
import pickle as pkl

def summarize_docs(docs: List[str]) -> str:
    '''
    This method is the abstraction that is provided to the end user for utilization. It outputs a summary
    for a given list of documents.
    '''
    max_output_sentences = 80

    sentence_mapper = None
    important_paragraphs = []
    faulty_mapper = {}

    start = time()
    logging.info('method: summarize_docs- Finding abbreviations in the text')
    abbrevs = find_abbreviations(docs)
    end = time()
    print('Time taken to find abbreviations: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Extracting top_k paragraphs from the document')
    for idx, doc in enumerate(docs):
        mapper, top_k = extract_top_k_paragraphs(doc, abbrevs)
        if idx == 1:
            faulty_mapper = mapper
        if sentence_mapper is None: sentence_mapper = mapper
        else: sentence_mapper.update(mapper)
        important_paragraphs.extend(top_k)
    end = time()
    print('Time taken to extract top k paragraphs: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Post Processing the output of gathering important paragraphs')
    final_doc, final_sentence_mapper = _process_output_imp_para(sentence_mapper, important_paragraphs)
    end = time()
    print('Time taken to post-process important paragraphs: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Starting clustering of sentences in important paragraphs')
    clusters, noisy_data = extract_clusters(final_doc)
    end = time()
    print('Time taken to cluster sentences: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Extracting a good path for each of the found cluster')
    cluster_paths = []
    for cluster in clusters:
        possible_path = extract_possible_path(cluster, final_sentence_mapper, abbrevs)
        mapper, processed_path = rigorous_pre_process(possible_path, abbrevs)
        cluster_paths.append(processed_path[0][0])
        final_sentence_mapper.update(mapper)
    end = time()
    print('Time taken to extract a good path from cluster: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Ranking sentences for preparing final summary')
    ranked_sentences = rank_sentences(cluster_paths, noisy_data)
    fraction_output = 0.4
    num_output_sentences = min(int(fraction_output * len(ranked_sentences)), max_output_sentences)
    output = []
    num_added = 0
    for sentence in ranked_sentences:
        if num_added == num_output_sentences: break
        output.append(final_sentence_mapper[sentence])
        num_added += 1
    end = time()
    print('Time taken to rank sentences: {:03f} seconds'.format(end - start))

    return mgr.sentence_separator.join(output)

def _process_output_imp_para(sentence_mapper: Dict[str, str], list_of_important_paragraphs: List[List[str]]) -> (List[List[str]], Dict[str, str]):
    final_mapper = {}

    for imp_paragraph in list_of_important_paragraphs:
        for sentence in imp_paragraph:
            final_mapper[sentence] = sentence_mapper[sentence]

    return list_of_important_paragraphs, final_mapper
