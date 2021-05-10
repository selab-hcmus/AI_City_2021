import nltk
nltk.download('wordnet')

import json
import re
from tqdm.notebook import tqdm
import pandas as pd
import random
import webcolors
from matplotlib.colors import is_color_like
from allennlp.predictors.predictor import Predictor
import Levenshtein
import spacy
from nltk.stem import WordNetLemmatizer

from srl_extractor import SRL

def main():
  train_path = './data/train-tracks.json'
  test_path = './data/test-queries.json'

  # Running on train queries
  srl_train = SRL(path=train_path, filetype=1)
  ans_train = srl_train.extract_data(srl_train.data)
  f = open('./results/result_train.json', 'w')
  json.dump(ans_train, f, indent=2)
  f.close()

  # Running on test queries
  srl_test = SRL(path=test_path, filetype=0)
  ans_test = srl_test.extract_data(srl_test.data)
  f = open('./results/result_test.json', 'w')
  json.dump(ans_test, f, indent=2)
  f.close()

if __name__ == "__main__":
  main()