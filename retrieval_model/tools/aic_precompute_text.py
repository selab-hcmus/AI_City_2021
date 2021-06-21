import sys
import json
import os
import shutil
import time
from copy import deepcopy
from itertools import permutations 
from timeit import default_timer as timer
from typing import Callable, Dict, List


import h5py
import numpy as np
import torch as th
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from retrieval_model import nntrainer
import retrieval_model.nntrainer.data_text
from retrieval_model.nntrainer import arguments, maths, utils
from retrieval_model.nntrainer.data_text import get_text_preprocessor
from retrieval_model.nntrainer.typext import ConstantHolder

from retrieval_model.coot.text_dataset import TextConverterDataset

from retrieval_model.utils.text_utils import (
    init_embedding_module, get_preprocessor, get_text_dict
)



def get_parser():
    parser = utils.ArgParser()
    parser.add_argument("--dataset_name", type=str, default='TRAIN', help="dataset name")
    arguments.add_dataset_path_arg(parser)
    arguments.add_test_arg(parser)
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--model_path", type=str, default=None, help="Cache path for transformers package.")
    parser.add_argument("--layers", type=str, default="-2,-1",
                        help="Read the features from these layers. Careful: Multiple layers must be specified like "
                             "this: --layers=-2,-1 because of argparse handling minus as new argument.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--add_name", type=str, default="", help="Add additional identifier to output files.")
    parser.add_argument("--encoder_only", action="store_true",
                        help="Flag for hybrid models (BART: bilinear and unilinear) that return "
                             "both encoder and decoder output, if the decoder output should be discarded.")
    parser.add_argument("--set_tokenizer", type=str, default="",
                        help=f"Manually define the tokenizer instead of determining it from model name. "
                             f"Options: {nntrainer.data_text.TextPreprocessing.values()}")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help=f"Set the tokenizer to add special tokens (like [CLS], [SEP] for BERT).")
    parser.add_argument("--token_stride", action="store_true",
                        help=f"If set, too long texts will be strided over instead of cut to max.")
    parser.add_argument("--token_stride_factor", type=int, default=2,
                        help=f"Default 2 means to stride half the window size. Set to 1 for non-overlapping windows.")
    parser.add_argument("--print_model", action="store_true", help=f"Print model and config")

    args = parser.parse_args()
    return args

def test_preprocess_text(text_dict, preprocessor):
    sample_id = list(text_dict.keys())[0]
    old_text = text_dict[sample_id]
    new_text = preprocessor(old_text)
    print('*'*20, 'old text')
    print(old_text)
    print('*'*20, 'new text')
    print(new_text)
    return 

def test_embedding_feature(text_dict, data_npy, length):
    sample_id = list(text_dict.keys())[0]
    print('Input text')
    print(text_dict[sample_id])
    print('Input length')
    print(length[sample_id])
    print(f'Embeding feature: {data_npy[sample_id].shape}')
    pass

@th.no_grad()
def main():
    args = get_parser()

    data_path = arguments.update_path_from_args(args)
    dataset_path = data_path #/ 'aic21'
    model_name = args.model_name
    token_stride = args.token_stride
    save_name = f"{args.dataset_name}_text_feat"
    # setup paths
    text_features_path = dataset_path
    os.makedirs(text_features_path, exist_ok=True)
    lengths_file = text_features_path / f"{save_name}_sentence_splits.json"
    data_file_only = f"{save_name}.h5"
    data_file = text_features_path / data_file_only

    if data_file.exists() and lengths_file.exists():
        print(f"{data_file} already exists. nothing to do.")
        return

    # Init model
    model, tokenizer = init_embedding_module(args)
    max_text_len = model.config.max_position_embeddings
    print('Init model successfully')

    # define preprocessor
    is_tp = False
    add_special_tokens = args.add_special_tokens
    preprocessor = get_preprocessor(args)

    # define feature layers to extract
    layer_list_int = [int(layer.strip()) for layer in args.layers.strip().split(",")]

    # load text_dict
    text_dict = get_text_dict(args)
    print('Load textual data successfully')

    # TEST preprocess text:
    # test_preprocess_text(text_dict, preprocessor)

    # get max number of words length
    total_words = 0
    max_words = 0
    for key, val in tqdm(text_dict.items(), desc="Compute total_words and max_words"):
        num_words = sum(len(text.split(" ")) for text in val)
        total_words += num_words
        max_words = max(num_words, max_words)
    print(f"Total {total_words} average {total_words / len(list(text_dict.keys())):.2f} max {max_words}")

    # create dataset and loader
    print("*" * 20, "Loading and testing dataset.")
    dataset = TextConverterDataset(tokenizer, text_dict, preprocessor, max_text_len=max_text_len, 
                                    token_stride=token_stride, add_special_tokens=add_special_tokens)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=dataset.collate_fn)

    # print first datapoint
    for key, value in dataset[0].items():
        print(f"{key}: {value}\n")

    # loop videos and encode features
    print("*" * 20, "Running the encoding.")
    print(f"Encoding text with model: {model_name}, layers: {layer_list_int}, "
          f"batch size: {args.batch_size}, workers: {args.workers}")
    temp_file = text_features_path / f"TEMP_{utils.get_timestamp_for_filename()}_{data_file_only}"
    data_h5 = h5py.File(temp_file, "w")
    data_npy = {}
    lengths = {}
    total_feat_dim = None
    printed_warning = False
    pbar = tqdm(desc="Compute text features", total=maths.ceil(len(dataset) / args.batch_size))
    for i, batch in enumerate(dataloader):  # type: TextDataBatchPoint
        if args.cuda:
            batch.to_cuda(non_blocking=True)
        batch_size = len(batch.key)

        total_max_seq_len = batch.tokens.shape[1]
        if total_max_seq_len <= max_text_len:
            # everything is fine
            # compute model output and read hidden states
            model_outputs = model(input_ids=batch.tokens, attention_mask=batch.mask, output_hidden_states=True)
            hidden_states = model_outputs["hidden_states"]
            # pbar.write(f"tokens {batch.tokens.shape[1]}")
            # pbar.write(f"outputs {list(state.shape[1] for state in hidden_states)}")
            # concatenate the features from the requested layers of the hidden state (-1 is the output layer)
            features = []
            for layer_num in layer_list_int:
                layer_features = hidden_states[layer_num]
                features.append(layer_features.detach().cpu().numpy())
            # concatenate features of individual hidden layers
            features = np.concatenate(features, axis=-1)  # shape (batch_size, max_sent_len, num_layers * feat_dim)
            # pbar.write(f"features {features.shape}")
        else:
            # if batch tokens is too long we need multiple steps depending on stride
            stride = max_text_len // args.token_stride_factor
            positions = list(range(0, total_max_seq_len - stride, stride))
            all_model_outputs = []
            pbar.write(f"Length {total_max_seq_len}! Split with window {max_text_len} stride {stride} "
                       f"into {len(positions)} batches at positions {positions} ")
            for pos in positions:
                end_pos = pos + max_text_len
                these_tokens = batch.tokens[:, pos:end_pos]
                these_masks = batch.mask[:, pos:end_pos]
                these_model_outputs = model(input_ids=these_tokens, attention_mask=these_masks,
                                            output_hidden_states=True)
                these_hidden_states = these_model_outputs["hidden_states"]
                # pbar.write(f"tokens {these_tokens.shape[1]}")
                # pbar.write(f"outputs {list(state.shape[1] for state in these_hidden_states)}")
                # concatenate the features from the requested layers of the hidden state (-1 is the output layer)
                features = []
                for layer_num in layer_list_int:
                    layer_features = these_hidden_states[layer_num]
                    if pos != 0:
                        layer_features = layer_features[:, stride:]
                    features.append(layer_features.detach().cpu().numpy())
                # concatenate features of individual hidden layers
                features = np.concatenate(features, axis=-1)  # shape (batch_size, max_sent_len, num_layers * feat_dim)
                # pbar.write(f"features {features.shape}")
                all_model_outputs.append(features)
            # concatenate outputs back together
            features = np.concatenate(all_model_outputs, axis=1)

        # compute total output size, need to know this for model architecture
        if total_feat_dim is None:
            total_feat_dim = features.shape[-1]

        # extract single datapoint information from the batch
        for batch_num in range(batch_size):
            key = batch.key[batch_num]
            length = batch.lengths[batch_num]

            # given length (number of tokens), cut off the padded tokens
            feature = features[batch_num, :length]

            # store sentence lengths so features can be mapped to sentences later
            sentence_lengths = batch.sentence_lengths[batch_num]

            if is_tp:
                sentence_lengths = [int(np.round(length / 4)) for length in sentence_lengths]

            # make sure correspondence between paragraph features and sentence lengths is still there
            if feature.shape[0] != sum(sentence_lengths) and not printed_warning:
                pbar.write("*" * 40)
                pbar.write(f"WARNING: Feature sequence length {feature.shape[0]} is not equal sum of the sentence "
                           f"lengths: "f"{sum(sentence_lengths)}")
                pbar.write(f"{sentence_lengths}")
                pbar.write(f"It may be hard to get the correspondence between tokens and features back and the "
                           f"correct hierarchical sentence structure back from these features..")
                printed_warning = True

            # write features
            data_npy[key] = feature
            data_h5[key] = feature
            lengths[key] = sentence_lengths
            
        pbar.update()
    pbar.close()
    data_h5.close()

    print(f"Wrote data to {temp_file}, moving to {data_file}")
    if data_file.is_file():
        os.remove(data_file)
        time.sleep(0.1)
    shutil.move(temp_file, data_file)

    # write lengths file
    json.dump(lengths, lengths_file.open("wt", encoding="utf8"))

    print(f"Wrote sentence splits to {lengths_file}")
    print(f"Total feature dim of {len(layer_list_int)} is {total_feat_dim}")


if __name__ == "__main__":
    main()
