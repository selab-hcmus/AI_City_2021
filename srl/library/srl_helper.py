import Levenshtein
import re

from .data_helper import DataHelper
from .srl_predictor import SRLPredictor

class SRLHelper(object):
  def __init__(self, filename=None, get_predictor=False):
    try:
      data_helper = DataHelper()
      data_info = data_helper.load_file(filename)
      if get_predictor:
        self.predictor = SRLPredictor(path=data_info['predictor'])
      else:
        self.predictor = None
      self.semantic_key_converter = data_helper.load_file(data_info['semantic_key_converter'])
      self.vehicle_converter = data_helper.load_file(data_info['vehicle_converter'])
      self.action_converter = data_helper.load_file(data_info['action_converter'])
      self.list_vehicle = data_helper.load_file(data_info['list_vehicle'])
      self.vehicle_vocab = data_helper.load_file(data_info['vehicle_vocab'])
      self.spelling_correction = data_helper.load_file(data_info['spelling_correction'])
    except:
      self.predictor = None
      self.semantic_key_converter = None
      self.vehicle_converter = None
      self.action_converter = None
      self.vehicle_vocab = None
      self.spelling_correction = None

  def refactor_vehicle(self, query):
    query = ' ' + query + ' '
    for val in self.vehicle_converter:
      target_word = val['target_word']
      new_word = ' ' + val['new_word'] + ' '
      while query.find(target_word) >= 0:
        query = query.replace(target_word, new_word)
    return query[1:len(query)-1]

  def refactor_action(self, query):
    query = ' ' + query + ' '
    for val in self.action_converter:
      target_words = val['target_word']
      word_to_skip = val['word_to_skip']
      new_word = val['new_word']
      for target_word in target_words:
        new_word = ' ' + new_word + ' '
        while query.find(target_word) >= 0:
          if word_to_skip is None:
            query = query.replace(target_word, new_word)
          else:
            flag = True
            for word in word_to_skip:
              if query.find(word) >= 0:
                flag = False
                break
            if flag:
              query = query.replace(target_word, new_word)
            else:
              break
    return query[1:len(query)-1]

  def contain_word(self, word):
    for key in self.spelling_correction:
      if word in self.spelling_correction[key]:
        return True
    return False
  
  def correct_spelling(self, query):
    single_words = query.split()
    query = ' ' + query + ' '
    for keyword in self.spelling_correction['vehicle_2']:
      while True:
        indices = [i for i, c in enumerate(query) if c == ' ']
        if len(indices) <= 3:
          break
        for i in range(len(indices)-2):
          word = query[indices[i]+1:indices[i+2]]
          if Levenshtein.distance(word, keyword) == 1:
            query = query[:indices[i]+1] + keyword + query[indices[i+2]:]
            break
        if i == len(indices)-3:
          break
    
    keys = ["color", "vehicle_1", "action"]
    skip_words = self.spelling_correction['skip_word']
    for key in keys:
      keywords = self.spelling_correction[key]
      for keyword in keywords:
        words = query.split()
        for i, word in enumerate(words):
          if self.contain_word(word) or keyword in word:
            continue
          elif Levenshtein.distance(word, keyword) == 1:
            words[i] = keyword
        query = ' '.join(words)
    return query

  def convert_keywords(self, query):
    dict_kw = {
        "without stopping": "",
        "not stopping": "",
        "make a turn": "turned",
        "makes a turn": "turned",
        " take a": " turned",
        " takes a": " turned",
    }

    for kw in dict_kw:
      while query.find(kw) >= 0:
        query = query.replace(kw, dict_kw[kw])
    return query 

  def clean_query_before_inp(self, query):
    query = query.lower()
    query = re.sub('[^a-z. -/]', '', query)
    query = self.correct_spelling(query)
    query = self.convert_keywords(query)
    query = self.refactor_vehicle(query)
    query = self.refactor_action(query)
    return ' '.join(query.split())

  def clean_query_after_out(self, query):
    inp_hyphen = ' - '
    out_hyphen = '-'
    while query.find(inp_hyphen) >= 0:
      query = query.replace(inp_hyphen, out_hyphen)
    query = re.sub(' +', ' ', query)
    return query