import sys, os
from copy import deepcopy
from timeit import default_timer as timer
from typing import Callable, Dict, List

sys.path.append(os.getcwd())

from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

import torch as th
from torch.utils import data
from nntrainer.typext import ConstantHolder, TypedNamedTuple


# ---------- Text Dataset ----------

class TextDataPointTuple(TypedNamedTuple):
    """
    Definition of a single hierarchical text datapoint.
    """
    key: str
    text: List[str]
    text_tokenized: List[List[str]]
    tokens: th.Tensor  # shape: (num_tokens)
    sentence_lengths: List[int]

class TextDataBatchPoint(TypedNamedTuple):
    """
    Definition of a hierarchical text batch.
    """
    key: List[str]
    tokens: th.Tensor  # shape: (batch_size, max_num_tokens)
    mask: th.BoolTensor  # shape: (batch_size, max_num_tokens)
    lengths: th.Tensor  # shape: (batch_size)
    sentence_lengths: List[List[int]]

class TextConverterDataset(data.Dataset):
    """
    Dataset used for text input to generate features with language models.

    Args:
        tokenizer: String to int tokenizer.
        text_dict: Input text dict, each value is a list of sentences.
        preprocessor: Preprocessing function for the text.
        max_text_len: Maximum input length for the model.
        min_cut: Minimum sentence length to retain when cutting input.
        add_special_tokens: Let the tokenizer add special tokens like [CLS].
            Only set True if the preprocessor doesn't do that already.
    """

    def __init__(self, tokenizer: BertTokenizer, text_dict: Dict[str, List[str]],
                 preprocessor: Callable[[List[str]], List[List[str]]], *, max_text_len: int = 512, min_cut: int = 5,
                 token_stride: bool = False,
                 add_special_tokens: bool = False):
        self.token_stride = token_stride
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.preprocessor = preprocessor
        self.max_text_len = max_text_len
        self.min_cut = min_cut
        self.keys = list(text_dict.keys())
        self.add_special_tokens = add_special_tokens

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, item: int) -> TextDataPointTuple:
        key: str = self.keys[item]
        text: List[str] = self.text_dict[key]

        # process paragraph text
        processed_text: List[str] = self.preprocessor(text)

        # tokenize with the model's tokenizer
        total_len: int = 0
        par_tokens: List[List[int]] = []
        par_tokens_str: List[int] = []

        for sentence in processed_text:
            sentence_tokens_str = self.tokenizer.tokenize(sentence, add_special_tokens=self.add_special_tokens)
            sentence_tokens = self.tokenizer.convert_tokens_to_ids(sentence_tokens_str)
            total_len += len(sentence_tokens)
            par_tokens.append(sentence_tokens)
            par_tokens_str.append(sentence_tokens_str)

        # check max length is fulfilled only if token_stride is disabled
        if sum(len(sentence_tokens) for sentence_tokens in par_tokens) > self.max_text_len and not self.token_stride:
            # automatically cut too long tokens if needed
            original_sentence_lengths = [len(sentence) for sentence in par_tokens]
            new_sentence_lengths = deepcopy(original_sentence_lengths)

            # go through sentences backwards and calculate new lengths
            for sent in reversed(range(len(new_sentence_lengths))):
                # calculate how much there is still left to cut
                overshoot = sum(new_sentence_lengths) - 512
                if overshoot <= 0:
                    break

                # don't cut more than min_cut
                new_len = max(self.min_cut, len(par_tokens[sent]) - overshoot)
                new_sentence_lengths[sent] = new_len

            # given the calculated new lengths, iterate sentences and make them shorter
            par_tokens_new = []
            for i, (old_len, new_len) in enumerate(zip(original_sentence_lengths, new_sentence_lengths)):
                if old_len == new_len:
                    # nothing changed, retain old sentence
                    par_tokens_new.append(par_tokens[i])
                    continue

                # cut the sentence to new length L, keep first L-1 and the last EOS token.
                par_tokens_new.append(par_tokens[i][:new_len - 1] + [par_tokens[i][-1]])

            # done, replace tokens
            par_tokens = par_tokens_new
            print(f"\nKey: {key}, Cut input {sum(original_sentence_lengths)} to {self.max_text_len}, new length: "
                  f"{sum(len(sentence) for sentence in par_tokens)}")

        # calculate sentence lengths, these are needed to get the features back per sentence
        sentence_lengths = [len(sentence) for sentence in par_tokens]

        # input an entire flat paragraph into the model to make use of context
        flat_tokens = th.Tensor([word for sentence in par_tokens for word in sentence]).long()
        return TextDataPointTuple(key, processed_text, par_tokens_str, flat_tokens, sentence_lengths)

    def collate_fn(self, batch: List[TextDataPointTuple]):
        """
        Collate a list of datapoints, merge tokens into a single tensor and create attention masks.

        Args:
            batch: List of single datapoints.

        Returns:
            Collated batch.
        """
        batch_size = len(batch)

        # get tokens and calculate their length
        list_tokens = [b.tokens for b in batch]
        list_lengths = [len(token) for token in list_tokens]
        lengths = th.Tensor(list_lengths).long()

        # initialize batch tensors to the max sequence length
        max_len = max(list_lengths)
        tokens = th.zeros(batch_size, max_len).long()
        mask = th.zeros(batch_size, max_len).bool()

        # given lengths and content, fill the batch tensors
        for batch_num, (seq_len, item) in enumerate(zip(list_lengths, list_tokens)):
            tokens[batch_num, :seq_len] = item
            mask[batch_num, :seq_len] = 1

        # add some more required information to the batch
        key = [b.key for b in batch]
        sentence_lengths = [b.sentence_lengths for b in batch]
        return TextDataBatchPoint(key, tokens, mask, lengths, sentence_lengths)
