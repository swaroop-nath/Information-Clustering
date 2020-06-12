from typing import List, Dict
from paragraph_ranking import extract_top_k_paragraphs
from sentence_clustering import extract_clusters
from text_utils import find_abbreviations, rigorous_pre_process
from sentence_ranking import rank_sentences
from chunk_graphs import extract_possible_path
import manager as mgr
from nltk import sent_tokenize
import logging
from time import time
import pickle as pkl

def summarize_docs(docs: List[str]) -> str:
    '''
    This method is the abstraction that is provided to the end user for utilization. It outputs a summary
    for a given list of documents.
    '''
    # sentence_mapper = None
    # important_paragraphs = []

    # start = time()
    # logging.info('method: summarize_docs- Finding abbreviations in the text')
    # abbrevs = find_abbreviations(docs)
    # end = time()
    # print('Time taken to find abbreviations: {:03f} seconds'.format(end - start))

    # start = time()
    # logging.info('method: summarize_docs- Extracting top_k paragraphs from the document')
    # for doc in docs:
    #     mapper, top_k = extract_top_k_paragraphs(doc, abbrevs, k = 50)
    #     if sentence_mapper is None: sentence_mapper = mapper
    #     else: sentence_mapper.update(mapper)
    #     important_paragraphs.append(top_k)
    # end = time()
    # print('Time taken to extract top k paragraphs: {:03f} seconds'.format(end - start))

    # start = time()
    # logging.info('method: summarize_docs- Post Processing the output of gathering important paragraphs')
    # final_doc, final_sentence_mapper = _process_output_imp_para(sentence_mapper, important_paragraphs)
    # del sentence_mapper, important_paragraphs
    # end = time()
    # print('Time taken to post-process important paragraphs: {:03f} seconds'.format(end - start))

    # start = time()
    # logging.info('method: summarize_docs- Starting clustering of sentences in important paragraphs')
    # clusters, noisy_data = extract_clusters(final_doc)
    # end = time()
    # print('Time taken to cluster sentences: {:03f} seconds'.format(end - start))

    # with open('clusters.dmp', 'wb') as file: pkl.dump(clusters, file)
    # with open('noise.dmp', 'wb') as file: pkl.dump(noisy_data, file)
    # with open('mapper.dmp', 'wb') as file: pkl.dump(final_sentence_mapper, file)

    with open('clusters.dmp', 'rb') as file: clusters = pkl.load(file)
    with open('noise.dmp', 'rb') as file: noisy_data = pkl.load(file)    
    with open('mapper.dmp', 'rb') as file: final_sentence_mapper = pkl.load(file)
    abbrevs = {'ANCYL': 'ANC Youth League', 'PAC': 'Pan Africanist Congress', 'ANC': 'African National Congress', 'TRC': 'Truth and Reconciliation Commission'}

    start = time()
    logging.info('method: summarize_docs- Extracting a good path for each of the found cluster')
    cluster_paths = []
    for cluster in clusters:
        possible_path = extract_possible_path(cluster, final_sentence_mapper, abbrevs)
        mapper, processed_path = rigorous_pre_process(possible_path, abbrevs)
        cluster_paths.append(processed_path)
        final_sentence_mapper.update(mapper)
    del clusters
    end = time()
    print('Time taken to extract a good path from cluster: {:03f} seconds'.format(end - start))

    start = time()
    logging.info('method: summarize_docs- Ranking sentences for preparing final summary')
    ranked_sentences = rank_sentences(cluster_paths, noisy_data)
    del cluster_paths, noisy_data
    fraction_output = 0.4
    num_output_sentences = int(fraction_output * len(ranked_sentences))
    output = []
    num_added = 0
    for sentence in ranked_sentences:
        if num_added == num_output_sentences: break
        output.append(sentence)
        num_added += 1
    del ranked_sentences
    end = time()
    print('Time taken to rank sentences: {:03f} seconds'.format(end - start))

    return mgr.sentence_separator.join(output)

def _process_output_imp_para(sentence_mapper: Dict[str, str], list_of_important_paragraphs: List[str]) -> (str, Dict[str, str]):
    doc = mgr.paragraph_separator.join(list_of_important_paragraphs)
    final_mapper = {}

    for imp_paragraphs in list_of_important_paragraphs:
        for paragraph in imp_paragraphs.split('\n\n'):
            for sentence in sent_tokenize(paragraph):
                final_mapper[sentence] = sentence_mapper[sentence]

    return doc, final_mapper
