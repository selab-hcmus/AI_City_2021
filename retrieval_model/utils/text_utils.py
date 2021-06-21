import sys
import json
import os

sys.path.append(os.getcwd())

import h5py
import numpy as np
import torch as th
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from nntrainer.data_text import get_text_preprocessor
from retrieval_model.coot.dataset_utils import (
    TRAIN_TRACK_JSON, TEST_QUERY_JSON
)
from utils.data_manager import TRAIN_TRACK_JSON, TEST_QUERY_JSON
CONFIG = {
    'model_source': 'transformers', 
    'model_name': "bert-base-uncased", 
    "model_path": None, 
    "layers": "-2,-1", 
}

class TextModelConst(ConstantHolder):
    """
    Identifier for text models, the model name starts with the identifier.
    """
    BERT = "bert"
    GPT2 = "gpt2"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"


def init_embedding_module(args):
    # Load pretrained model
    model_name = args.model_name
    print("*" * 20, f"Loading model {model_name} from {args.model_source}")
    if args.model_source == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.model_path)
        model: BertModel = AutoModel.from_pretrained(model_name, cache_dir=args.model_path)

        # noinspection PyUnresolvedReferences
        max_text_len = model.config.max_position_embeddings
        model.eval()
    else:
        raise NotImplementedError(f"Model source unknown: {args.model_source}")
    if args.cuda:
        model = model.cuda()

    print(f"Running model on device {next(model.parameters()).device}")
    print(f"Maximum input length {max_text_len}")

    return model, tokenizer

def get_preprocessor(args):
    model_name = CONFIG['model_name']
    preprocessor = None
    if args.set_tokenizer != "":
        print(f"Set tokenizer via flag to {args.set_tokenizer}")
        preprocessor = get_text_preprocessor(args.set_tokenizer)
    elif model_name == "bert-base-uncased":
        # paper results
        preprocessor = get_text_preprocessor(nntrainer.data_text.TextPreprocessing.BERT_PAPER)
    elif model_name.startswith(TextModelConst.BERT) or model_name.startswith(TextModelConst.DISTILBERT):
        # new results bert-large-cased
        preprocessor = get_text_preprocessor(nntrainer.data_text.TextPreprocessing.BERT_NEW)
    elif model_name.startswith(TextModelConst.GPT2):
        # new results with gpt2
        preprocessor = get_text_preprocessor(nntrainer.data_text.TextPreprocessing.GPT2)
    else:
        print(f"WARNING: no text preprocessing defined for model {model_name}, using default preprocessing which "
              f"does not add any special tokens.")
        preprocessor = get_text_preprocessor(nntrainer.data_text.TextPreprocessing.SIMPLE)

    return preprocessor

def get_text_dict(args):
    text_dict = {} # {'query_id': [cap1, cap2, cap3]}
    if args.dataset_name == 'TEST':
        meta_json = TEST_QUERY_JSON
        test_queries = json.load(open(meta_json))
        for q_id in test_queries:
            text_dict[q_id] = test_queries[q_id]    
    else:
        meta_json = TRAIN_TRACK_JSON

        train_track = json.load(open(meta_json))
        for q_id in train_track:
            text_dict[q_id] = train_track[q_id]['nl']

    return text_dict
