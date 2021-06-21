import h5py, os, json, sys, shutil, pickle
import numpy as np 
from tqdm import tqdm
import os.path as osp
sys.path.append(os.getcwd())

import torch
from torch.cuda.amp import autocast
from torch import nn

from retrieval_model.coot.configs_retrieval import ExperimentTypesConst, RetrievalConfig as Config
from retrieval_model.coot.aic_dataset import create_retrieval_datasets_and_loaders, RetrievalDataset
from retrieval_model.coot.test_dataset import VideoDataset, TextDataset
from retrieval_model.coot.model_retrieval import RetrievalModelManager as ModelManager
from retrieval_model.coot.trainer_retrieval import RetrievalTrainer as Trainer

from retrieval_model.nntrainer import arguments, utils
from retrieval_model.nntrainer.utils_torch import set_seed
from retrieval_model.nntrainer.utils_yaml import load_yaml_config_file
from retrieval_model.nntrainer import data as nn_data, data_text, maths, typext, utils, utils_torch
from retrieval_model.nntrainer import retrieval


EXP_TYPE = ExperimentTypesConst.RETRIEVAL

def setup_config():
    # ---------- Setup script arguments. ----------
    parser = utils.ArgParser(description=__doc__)
    arguments.add_default_args(parser)  # logging level etc.
    arguments.add_exp_identifier_args(parser)  # arguments to identify the experiment to run
    arguments.add_trainer_args(parser)  # general trainer arguments
    arguments.add_dataset_test_arg(parser)  # flag for dataset testing

    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--save_embeddings", action="store_true", help="Save generated COOT embeddings.")
    parser.add_argument("--save_name", type=str, default='default_submission', help="Load model from file.")
    parser.add_argument("--vid_feat", type=str, default="test_resnet152-224_6148", help="Load model from file.")
    
    args = parser.parse_args()

    if args.save_embeddings:
        assert args.validate, "Saving embeddings only works in validation with --validate"

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config and dataset path given the script arguments
    path_data = arguments.update_path_from_args(args)
    config = arguments.update_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config, is_train=not args.validate and not args.test_dataset)

    return cfg, path_data, args, exp_group, exp_name

def setup_environment(cfg):
    # set seed
    if cfg.random_seed is not None:
        print(f"Set seed to {cfg.random_seed}")
        set_seed(cfg.random_seed, set_deterministic=False)  # set deterministic via config if needed

    # create dataset and dataloader
    if (cfg.dataset_train.preload_vid_feat or cfg.dataset_train.preload_text_feat or cfg.dataset_val.preload_vid_feat or
            cfg.dataset_val.preload_text_feat):
        cmd = "ulimit -n 100000"
        print(f"Run system command to avoid TooManyFiles error:\n{cmd}")
        os.system(cmd)

    pass

def normalize_emb(data_collector, collect_keys):
    data_collector_norm = {}
    for key in collect_keys:
        data_collector[key] = torch.cat(data_collector[key], dim=0).float()
        # data_collector_norm[key] = F.normalize(data_collector[key])
        data_collector_norm[key] = data_collector[key] / (data_collector[key] * data_collector[key]).sum(
            dim=-1).sqrt().unsqueeze(-1)
    
    return data_collector_norm

def move_model_to_device(model_mgr, cfg):
    for model in model_mgr.model_dict.values():
        if cfg.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
        
    pass

@torch.no_grad()
def inference_on_val():
    cfg, path_data, args, exp_group, exp_name = setup_config()
    setup_environment(cfg)

    print(f'path_data: {path_data}')

    print('*'*20 + 'Init dataset')
    test_set = RetrievalDataset(cfg.dataset_val, path_data)
    test_loader = nn_data.create_loader(
        test_set, cfg.dataset_val, cfg.val.batch_size, collate_fn=test_set.collate_fn
    )

    print()
    print('*'*20 + 'Create Model')    
    # create coot models
    model_mgr = ModelManager(cfg)
    model_mgr.set_model_state(torch.load(str(args.load_model)))
    move_model_to_device(model_mgr, cfg)
    load_best=False
    
    print()
    print('*'*20 + 'Start Inference')
    collect_keys = ["vid_context", "par_context"]
    data_collector = {}

    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch.to_cuda(non_blocking = cfg.cuda_non_blocking)

        with autocast(enabled=cfg.fp16_val):
            visual_data = model_mgr.encode_visual(batch)
            text_data = model_mgr.encode_text(batch)
        
        all_data = {**visual_data.dict(), **text_data.dict()}
        for key in collect_keys:
            emb = all_data.get(key)
            if data_collector.get(key) is None:
                data_collector[key] = [emb.data.cpu()]
            else:
                data_collector[key] += [emb.data.cpu()]

    data_collector_norm = normalize_emb(data_collector, collect_keys)
    res_v2p, res_p2v, sum_vp_at_1, str_vp = retrieval.compute_retrieval(
            data_collector_norm, "vid_context", "par_context", print_fn=None)

    print(f'Vid-Par result: {res_v2p}')
    print(f'Par-Vid result: {res_p2v}')

def get_emb_matrix(data_collector, is_norm=True):
    keys = list(data_collector.keys())
    list_emb = []
    for k in keys:
        list_emb.append(data_collector[k]) 

    emb_mat = torch.cat(list_emb, dim=0).cpu().float()
    if is_norm:
        emb_mat = emb_mat/(emb_mat*emb_mat).sum(dim=-1).sqrt().unsqueeze(-1)

    return emb_mat

@torch.no_grad()
def inference_on_test():
    cfg, path_data, args, exp_group, exp_name = setup_config()
    setup_environment(cfg)
    model_id = args.load_model.split('/')[-3]
    print(f'path_data: {path_data}')

    print('*'*20 + 'Init dataset')
    
    cfg.dataset_val.vid_feat_name = args.vid_feat
    cfg.dataset_val.text_feat_name = "text_feat_TEST_meta_all_transformers_bert-base-uncased_-2,-1"

    video_set = VideoDataset(cfg.dataset_val, path_data) 
    text_set = TextDataset(cfg.dataset_val, path_data)
    video_loader = nn_data.create_loader(
        video_set, cfg.dataset_val, batch_size=1, collate_fn=video_set.collate_fn
    )
    text_loader = nn_data.create_loader(
        text_set, cfg.dataset_val, batch_size=1, collate_fn=text_set.collate_fn
    )
    video_keys, text_keys = video_set.keys, text_set.keys

    print()
    print('*'*20 + 'Create Model')    
    # create coot models
    model_mgr = ModelManager(cfg)
    model_mgr.set_model_state(torch.load(str(args.load_model)))
    move_model_to_device(model_mgr, cfg)
    model_mgr.set_all_models_eval()
    

    save_emb_dir = './results'
    exp_id = 'embedding'
    save_emb_dir = osp.join(save_emb_dir, exp_id)
    os.makedirs(save_emb_dir, exist_ok=True)
    print()
    print('*'*20 + 'Start Inference')
    collect_keys = ["vid_context", "par_context"]

    # ----------------------------------------------
    # Embedding video
    print('EMBEDDING VIDEO')
    video_collector = {}
    for step, batch in tqdm(enumerate(video_loader), total=len(video_loader)):
        batch.to_cuda(non_blocking = cfg.cuda_non_blocking)
        key = batch.key[0]

        with autocast(enabled=cfg.fp16_val):
            visual_data = model_mgr.encode_visual(batch)
        
        all_data = {**visual_data.dict()}
        emb = all_data.get('vid_context')
        video_collector[key] = emb
    
    save_path = osp.join(save_emb_dir, 'video_emb.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(video_collector, f)
    print(f'save video emb to {save_path}')
    vid_emb_mat = get_emb_matrix(video_collector, is_norm=True)

    # ----------------------------------------------
    # Embedding text
    print('EMBEDDING TEXT')
    text_collector = {}
    for step, batch in tqdm(enumerate(text_loader), total=len(text_loader)):
        batch.to_cuda(non_blocking = cfg.cuda_non_blocking)
        key = batch.key[0]
        with autocast(enabled=cfg.fp16_val):
            text_data = model_mgr.encode_text(batch)
        
        all_data = {**text_data.dict()}
        emb = all_data.get('par_context')
        text_collector[key] = emb.data.cpu()
        
    save_path = osp.join(save_emb_dir, 'text_emb.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(text_collector, f)
    print(f'save text emb to {save_path}')

    par_emb_mat = get_emb_matrix(text_collector, is_norm=True)

    # Get top similar tracks for each query
    print('Get final results')
    par_emb_mat = par_emb_mat.detach().numpy()
    vid_emb_mat = vid_emb_mat.detach().numpy()
    print(f'par_emb_mat shape: {par_emb_mat.shape}')
    print(f'vid_emb_mat shape: {vid_emb_mat.shape}')
    par2vid_sim = np.dot(par_emb_mat, vid_emb_mat.T)
    vid_keys = list(video_collector.keys())
    result = {}
    for i, key in tqdm(enumerate(text_collector.keys())):
        inds = np.argsort(par2vid_sim[i])[::-1]
        result[key] = [vid_keys[idx] for idx in inds]
    
    save_path = osp.join('./results', f'{args.save_name}.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'save submission file to {save_path}')


if __name__ == '__main__':
    inference_on_test()