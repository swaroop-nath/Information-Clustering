import string
from ast import literal_eval
from pycontractions import Contractions
import spacy
import neuralcoref
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import sent_tokenize

punkt_param = PunktParameters()
illegal_sentence_splitters = ['rev', 'ed', 'st', 'eg', 'vs', 'v', 'yr', 'rep', 'ltd', 'r.k', 'e.f', 'd', 'g.f', 'dec', 'c.v', 'w', 'a.c', 'feb', 'r.j', 'a.g', 'cie', 'h', 'nov', 'a.h', 'n.j', 'b.v', 's.g', 'dr', 'tues', 'w.w', 'w.va', 'l', 's.s', 'ill', 'l.f', 'ariz', 'n', 'u.n', 'n.v', 'j.b', 'conn', 'maj', 'bros', 'okla', 'p.a.m', 'f', 'e.m', 'd.w', 'a.t', 'a.m', 'm.d.c', 'vt', 'mrs', 'g', 'u.k', 'e', 'r', 'ore', 'p', 'ga', 'j.k', 'col', 'n.h', 'j.p', 'c.i.t', 'f.j', 'sept', 't', 'b.f', 'aug', 'minn', 'm', 'p.m', 'co', 'e.l', 'mich', 'g.d', 'h.c', 'wed', 'r.h', 'ph.d', 'a.s', 'tenn', 'ct', 'calif', 'i.m.s', 'ala', 'm.b.a', 'j.c', 'h.f', 'sep', 'r.t', 'r.a', 's.p.a', 'c', 'pa', 'oct', 'cos', 'fla', 'm.j', 'w.c', 'jr', 'sw', 'a.m.e', 'wash', 'gen', 'd.c', '. . ', 'kan', 'u.s.s.r', 'lt', 'wis', 'jan', 's.a', 'colo', 'sr', 'j.r', 's.c', 'u.s', 'n.m', 'u.s.a', 'g.k', 'n.c', 't.j', 'n.y', 'k', 'messrs', 'ky', 'a.a', 'h.m', 'ok', 'r.i', 's', 'ms', 'e.h', 'ft', 'l.a', 'n.d', 'j.j', 'ave', 'prof', 'chg', 'f.g', 'l.p', 'mr', 'st', 'sen', 'fri', 'nev', 'a.d', 'c.o.m.b', 'd.h', 'w.r', 'adm', 'reps', 'inc', 'va', 'mg', 'corp', 's.a.y']
punkt_param.abbrev_types = set(illegal_sentence_splitters)
sentence_tokenizer = PunktSentenceTokenizer(punkt_param)

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
period_regex = '\.'
valid_eos_token = '[!?]'

expander = Contractions(api_key = 'glove-wiki-gigaword-50')
assert list(expander.expand_texts(['loader_demo_text']))[0] == 'loader_demo_text'

spacy_tool = spacy.load('en_md')
neuralcoref.add_to_pipe(spacy_tool)
