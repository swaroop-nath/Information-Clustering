import string
from ast import literal_eval
from pycontractions import Contractions
import spacy
import neuralcoref

quotation_regex = '[“”""]'
accepted_apostrophe_symbols = ['’', '\'']
reference_regex = '[\[\(][\d]{1,3}[\]\)]'
allowed_printable_chars = set(string.printable)
acronym_period_regex = '(?<!\w)[A-Za-z]\.[A-Za-z]\.[A-Za-z]?\.?[A-Za-z]?\.?[A-Za-z]?\.?[A-Za-z]?\.?'
prefix = '[\[\(\{/\w]+'
suffix = '[\]\)\}/\w]+'

with open('abbreviations.mapper', 'r') as file:
    content = file.read()
    abbreviations_map = literal_eval(content)

paragraph_separator = '\n\n'
sentence_separator = ' '
token_separator = ' '
unnecessary_identifier_regex = '[0-9\[\]%/,()–\'<>^~`@|#$+:;’]'
unnecessary_space = '  '
unnecessary_unresolved_pron = '-PRON-'
unnecessary_apostrophe = ' \''
unnecessary_space_period = ' \.'

expander = Contractions(api_key = 'glove-wiki-gigaword-50')
list(expander.expand_texts(['loader_demo_text']))

spacy_tool = spacy.load('en_md')
neuralcoref.add_to_pipe(spacy_tool)
