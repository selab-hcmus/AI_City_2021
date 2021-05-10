import h5py, json, shutil
import numpy as np
import torch as th
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

import nntrainer.data_text
from nntrainer import arguments, maths, utils
from nntrainer.data_text import get_text_preprocessor
from nntrainer.typext import ConstantHolder, TypedNamedTuple

from precompute_text import (
    TextConverterDataset
)

@th.no_grad()
def main():
    parser = utils.ArgParser()
    parser.add_argument("dataset_name", type=str, help="dataset name")
    arguments.add_dataset_path_arg(parser)
    arguments.add_test_arg(parser)
    parser.add_argument("--metadata_name", type=str, default="all", help="change which metadata to load")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--multi_gpu", action="store_true", help="use multiple gpus")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Cache path for transformers package.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Which model to use.")
    parser.add_argument("--model_source", type=str, default="transformers", help="Where to get the models from.")
    parser.add_argument("--layers", type=str, default="-2,-1",
                        help="Read the features from these layers. Careful: Multiple layers must be specified like "
                             "this: --layers=-2,-1 because of argparse handling minus as new argument.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--add_name", type=str, default="", help="Add additional identifier to output files.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite embedding if exists.")
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
    data_path = arguments.update_path_from_args(args)
    dataset_path = data_path / args.dataset_name
    model_name = args.model_name
    token_stride = args.token_stride
    model_ident = f"{args.model_source}_{model_name.replace('/', '--')}_{args.layers}"
    mode = 'TEST'
    full_ident = f"text_feat_{args.dataset_name}_{mode}_{args.metadata_name}_{model_ident}{args.add_name}"

    # setup paths
    text_features_path = dataset_path
    os.makedirs(text_features_path, exist_ok=True)
    lengths_file = text_features_path / f"{full_ident}_sentence_splits.json"
    data_file_only = f"{full_ident}.h5"
    data_file = text_features_path / data_file_only

    if data_file.exists() and lengths_file.exists() and not args.force:
        print(f"{data_file} already exists. nothing to do.")
        return

    # Load pretrained model
    print("*" * 20, f"Loading model {model_name} from {args.model_source}")
    if args.model_source == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.model_path)
        model: BertModel = AutoModel.from_pretrained(model_name, cache_dir=args.model_path)
        if args.print_model:
            print("*" * 40, "Model")
            print(f"{model}")
            print("*" * 40, "Config")
            print(model.config)
        # noinspection PyUnresolvedReferences
        max_text_len = model.config.max_position_embeddings
        model.eval()
    else:
        raise NotImplementedError(f"Model source unknown: {args.model_source}")
    if args.cuda:
        if args.multi_gpu:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    print(f"Running model on device {next(model.parameters()).device}")
    print(f"Maximum input length {max_text_len}")

    # define preprocessor
    is_tp = False
    add_special_tokens = args.add_special_tokens
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
    # else:
    #     raise NotImplementedError(f"No preprocessing defined for model {model_name}")

    # define feature layers to extract
    layer_list_int = [int(layer.strip()) for layer in args.layers.strip().split(",")]

    # load metadata
    meta_file = dataset_path / f"meta_{args.metadata_name}.json"
    print(f"Loading meta file of {meta_file.stat().st_size // 1024 ** 2:.0f} MB")
    timer_start = timer()
    meta_dict = json.load(meta_file.open("rt", encoding="utf8"))
    print(f"Took {timer() - timer_start:.1f} seconds for {len(meta_dict)}.")
    text_dict: Dict[str, List[str]] = {}
    for key, meta in meta_dict.items():
        text_dict[key] = [seg["text"] for seg in meta["segments"]]
    # get max number of words length
    total_words = 0
    max_words = 0
    for key, val in tqdm(text_dict.items(), desc="Compute total_words and max_words"):
        num_words = sum(len(text.split(" ")) for text in val)
        total_words += num_words
        max_words = max(num_words, max_words)
    print(f"Total {total_words} average {total_words / len(meta_dict):.2f} max {max_words}")

    # create dataset and loader
    print("*" * 20, "Loading and testing dataset.")
    dataset = TextConverterDataset(tokenizer, text_dict, preprocessor, max_text_len=max_text_len,
                                   token_stride=token_stride,
                                   add_special_tokens=add_special_tokens)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 collate_fn=dataset.collate_fn)

    # print first datapoint
    for key, value in dataset[0].items():
        print(f"{key}: {value}\n")

    if args.test:
        # print first datapoint
        for point in dataset:
            for key, value in dict(point).items():
                print(f"{key}: {value}\n")
        print("Test, stopping here.")
        return

    # loop videos and encode features
    print("*" * 20, "Running the encoding.")
    print(f"Encoding text with model: {model_name}, layers: {layer_list_int}, "
          f"batch size: {args.batch_size}, workers: {args.workers}")
    temp_file = text_features_path / f"TEMP_{utils.get_timestamp_for_filename()}_{data_file_only}"
    data_h5 = h5py.File(temp_file, "w")
    lengths = {}
    total_feat_dim = None
    printed_warning = False
    pbar = tqdm(desc="compute text features", total=maths.ceil(len(dataset) / args.batch_size))
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